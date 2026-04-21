import json
import PyPDF2

try:
    from .utils import get_number_of_pages, remove_fields, llm_completion, extract_json
except ImportError:
    from utils import get_number_of_pages, remove_fields, llm_completion, extract_json


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_pages(pages: str) -> list[int]:
    """Parse a pages string like '5-7', '3,8', or '12' into a sorted list of ints."""
    result = []
    for part in pages.split(','):
        part = part.strip()
        if '-' in part:
            start, end = int(part.split('-', 1)[0].strip()), int(part.split('-', 1)[1].strip())
            if start > end:
                raise ValueError(f"Invalid range '{part}': start must be <= end")
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def _count_pages(doc_info: dict) -> int:
    """Return total page count for a PDF document."""
    if doc_info.get('page_count'):
        return doc_info['page_count']
    if doc_info.get('pages'):
        return len(doc_info['pages'])
    return get_number_of_pages(doc_info['path'])


def _get_pdf_page_content(doc_info: dict, page_nums: list[int]) -> list[dict]:
    """Extract text for specific PDF pages (1-indexed). Prefer cached pages, fallback to PDF."""
    cached_pages = doc_info.get('pages')
    if cached_pages:
        page_map = {p['page']: p['content'] for p in cached_pages}
        return [
            {'page': p, 'content': page_map[p]}
            for p in page_nums if p in page_map
        ]
    path = doc_info['path']
    with open(path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        total = len(pdf_reader.pages)
        valid_pages = [p for p in page_nums if 1 <= p <= total]
        return [
            {'page': p, 'content': pdf_reader.pages[p - 1].extract_text() or ''}
            for p in valid_pages
        ]


def _get_md_page_content(doc_info: dict, page_nums: list[int]) -> list[dict]:
    """
    For Markdown documents, 'pages' are line numbers.
    Find nodes whose line_num falls within [min(page_nums), max(page_nums)] and return their text.
    """
    min_line, max_line = min(page_nums), max(page_nums)
    results = []
    seen = set()

    def _traverse(nodes):
        for node in nodes:
            ln = node.get('line_num')
            if ln and min_line <= ln <= max_line and ln not in seen:
                seen.add(ln)
                results.append({'page': ln, 'content': node.get('text', '')})
            if node.get('nodes'):
                _traverse(node['nodes'])

    _traverse(doc_info.get('structure', []))
    results.sort(key=lambda x: x['page'])
    return results


# ── Tool functions ────────────────────────────────────────────────────────────

def get_document(documents: dict, doc_id: str) -> str:
    """Return JSON with document metadata: doc_id, doc_name, doc_description, type, status, page_count (PDF) or line_count (Markdown)."""
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({'error': f'Document {doc_id} not found'})
    result = {
        'doc_id': doc_id,
        'doc_name': doc_info.get('doc_name', ''),
        'doc_description': doc_info.get('doc_description', ''),
        'type': doc_info.get('type', ''),
        'status': 'completed',
    }
    if doc_info.get('type') == 'pdf':
        result['page_count'] = _count_pages(doc_info)
    else:
        result['line_count'] = doc_info.get('line_count', 0)
    return json.dumps(result)


def get_document_structure(documents: dict, doc_id: str) -> str:
    """Return tree structure JSON with text fields removed (saves tokens)."""
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({'error': f'Document {doc_id} not found'})
    structure = doc_info.get('structure', [])
    structure_no_text = remove_fields(structure, fields=['text'])
    return json.dumps(structure_no_text, ensure_ascii=False)


def get_page_content(documents: dict, doc_id: str, pages: str) -> str:
    """
    Retrieve page content for a document.

    pages format: '5-7', '3,8', or '12'
    For PDF: pages are physical page numbers (1-indexed).
    For Markdown: pages are line numbers corresponding to node headers.

    Returns JSON list of {'page': int, 'content': str}.
    """
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({'error': f'Document {doc_id} not found'})

    try:
        page_nums = _parse_pages(pages)
    except (ValueError, AttributeError) as e:
        return json.dumps({'error': f'Invalid pages format: {pages!r}. Use "5-7", "3,8", or "12". Error: {e}'})

    try:
        if doc_info.get('type') == 'pdf':
            content = _get_pdf_page_content(doc_info, page_nums)
        else:
            content = _get_md_page_content(doc_info, page_nums)
    except Exception as e:
        return json.dumps({'error': f'Failed to read page content: {e}'})

    return json.dumps(content, ensure_ascii=False)


def search_document(documents: dict, doc_id: str, query: str, model: str = None) -> str:
    """
    Search for relevant content in a document using top-down tree traversal.

    At each level of the tree, an LLM selects which nodes are relevant to the
    query based on their titles and summaries. The search drills down to the
    relevant leaf nodes and returns only their page content — never the whole
    document.

    Returns JSON list of {'page': int, 'content': str}.
    """
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({'error': f'Document {doc_id} not found'})

    structure = doc_info.get('structure', [])
    if not structure:
        return json.dumps({'error': 'Document has no structure'})

    def _select_relevant_indices(nodes: list, query: str) -> list:
        """Ask the LLM which nodes at this level are relevant to the query."""
        node_summaries = [
            {
                'index': i,
                'title': node.get('title', ''),
                'summary': node.get('summary', node.get('prefix_summary', '')),
            }
            for i, node in enumerate(nodes)
        ]

        prompt = (
            "Given the user query and the following document sections (with titles and "
            "summaries), select which sections are relevant to answering the query.\n\n"
            f"Query: {query}\n\n"
            f"Sections:\n{json.dumps(node_summaries, indent=2, ensure_ascii=False)}\n\n"
            "Return a JSON array of indices of the relevant sections. "
            "Only include sections that are clearly relevant. "
            "Example: [0, 2] or [] if none are relevant.\n"
            "Directly return the JSON array. Do not output anything else."
        )

        response = llm_completion(model=model, prompt=prompt)
        try:
            indices = extract_json(response)
            if isinstance(indices, list):
                return [i for i in indices if isinstance(i, int) and 0 <= i < len(nodes)]
        except Exception:
            pass
        # Fallback: treat all nodes as relevant
        return list(range(len(nodes)))

    def _collect_all_leaf_pages(node: dict) -> list:
        """Return page specs for every leaf under node (used as fallback)."""
        children = node.get('nodes', [])
        if not children:
            return [_leaf_page_spec(node)]
        result = []
        for child in children:
            result.extend(_collect_all_leaf_pages(child))
        return result

    def _leaf_page_spec(node: dict) -> dict:
        """Return a dict describing the page range of a leaf node."""
        return {
            'start': node.get('start_index') or node.get('line_num'),
            'end': node.get('end_index') or node.get('line_num'),
            'type': 'line' if node.get('line_num') is not None and node.get('start_index') is None else 'page',
        }

    def _traverse(nodes: list, query: str) -> list:
        """Recursively traverse the tree, collecting page specs for relevant leaves."""
        relevant_indices = _select_relevant_indices(nodes, query)
        leaf_specs = []
        for i in relevant_indices:
            node = nodes[i]
            children = node.get('nodes', [])
            if children:
                child_specs = _traverse(children, query)
                if child_specs:
                    leaf_specs.extend(child_specs)
                else:
                    # No child was selected — fall back to all leaves under this node
                    leaf_specs.extend(_collect_all_leaf_pages(node))
            else:
                leaf_specs.append(_leaf_page_spec(node))
        return leaf_specs

    leaf_specs = _traverse(structure, query)

    if not leaf_specs:
        return json.dumps([])

    # Collect unique page / line numbers
    page_nums = sorted({
        p
        for spec in leaf_specs
        if spec.get('start') is not None and spec.get('end') is not None
        for p in range(spec['start'], spec['end'] + 1)
    })

    if not page_nums:
        return json.dumps([])

    try:
        if doc_info.get('type') == 'pdf':
            content = _get_pdf_page_content(doc_info, page_nums)
        else:
            content = _get_md_page_content(doc_info, page_nums)
    except Exception as e:
        return json.dumps({'error': f'Failed to read page content: {e}'})

    return json.dumps(content, ensure_ascii=False)
