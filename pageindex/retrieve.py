import json
import logging
import math
import PyPDF2

try:
    from .utils import (
        get_number_of_pages,
        remove_fields,
        llm_completion,
        extract_json,
        embedding_completion,
    )
except ImportError:
    from utils import (
        get_number_of_pages,
        remove_fields,
        llm_completion,
        extract_json,
        embedding_completion,
    )


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


def _get_pdf_page_map(doc_info: dict) -> dict[int, str]:
    cached_pages = doc_info.get('pages')
    if cached_pages:
        return {p['page']: p['content'] for p in cached_pages}
    path = doc_info['path']
    with open(path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        return {
            i: pdf_reader.pages[i - 1].extract_text() or ''
            for i in range(1, len(pdf_reader.pages) + 1)
        }


def _collect_md_line_text(nodes: list, line_map: dict[int, str]) -> None:
    for node in nodes:
        line_num = node.get('line_num')
        if line_num is not None and line_num not in line_map:
            line_map[line_num] = node.get('text', '')
        if node.get('nodes'):
            _collect_md_line_text(node['nodes'], line_map)


def _get_md_line_map(doc_info: dict) -> dict[int, str]:
    line_map = {}
    _collect_md_line_text(doc_info.get('structure', []), line_map)
    return line_map


def _iter_leaf_nodes(nodes: list) -> list[dict]:
    leaf_nodes = []
    for node in nodes:
        children = node.get('nodes', [])
        if children:
            leaf_nodes.extend(_iter_leaf_nodes(children))
        else:
            leaf_nodes.append(node)
    return leaf_nodes


def _leaf_page_spec(node: dict) -> dict:
    """Return a dict describing the page range of a leaf node."""
    is_line_based_node = node.get('line_num') is not None and node.get('start_index') is None
    return {
        'start': node.get('start_index') or node.get('line_num'),
        'end': node.get('end_index') or node.get('line_num'),
        'type': 'line' if is_line_based_node else 'page',
    }


def _get_content_from_page_spec(doc_info: dict, spec: dict, pdf_page_map: dict[int, str] | None = None, md_line_map: dict[int, str] | None = None) -> str:
    start = spec.get('start')
    end = spec.get('end')
    if start is None or end is None:
        return ""
    if doc_info.get('type') == 'pdf':
        pdf_page_map = pdf_page_map or _get_pdf_page_map(doc_info)
        return "\n\n".join(
            pdf_page_map[p]
            for p in range(start, end + 1)
            if p in pdf_page_map and pdf_page_map[p]
        )
    md_line_map = md_line_map or _get_md_line_map(doc_info)
    return "\n\n".join(
        md_line_map[line]
        for line in range(start, end + 1)
        if line in md_line_map and md_line_map[line]
    )


def _build_embedding_text(node: dict, content: str) -> str:
    parts = [
        node.get('title', ''),
        node.get('summary') or node.get('prefix_summary', ''),
        content,
    ]
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def build_embedding_index(doc_info: dict, embedding_model: str) -> dict:
    """Build and cache an embedding index for the document's leaf nodes."""
    if not embedding_model:
        raise ValueError("embedding_model is required")

    embedding_indexes = doc_info.setdefault('embedding_indexes', {})
    existing = embedding_indexes.get(embedding_model)
    if existing and existing.get('items'):
        return existing

    structure = doc_info.get('structure', [])
    leaf_nodes = _iter_leaf_nodes(structure)
    pdf_page_map = _get_pdf_page_map(doc_info) if doc_info.get('type') == 'pdf' else None
    md_line_map = _get_md_line_map(doc_info) if doc_info.get('type') != 'pdf' else None

    items = []
    embedding_texts = []
    for node in leaf_nodes:
        spec = _leaf_page_spec(node)
        content = _get_content_from_page_spec(doc_info, spec, pdf_page_map=pdf_page_map, md_line_map=md_line_map)
        embedding_text = _build_embedding_text(node, content)
        if not embedding_text:
            continue
        items.append({
            'node_id': node.get('node_id'),
            'title': node.get('title', ''),
            'summary': node.get('summary') or node.get('prefix_summary', ''),
            'start': spec.get('start'),
            'end': spec.get('end'),
            'type': spec.get('type'),
        })
        embedding_texts.append(embedding_text)

    if not items:
        index = {'model': embedding_model, 'items': []}
        embedding_indexes[embedding_model] = index
        return index

    vectors = embedding_completion(model=embedding_model, input_texts=embedding_texts)
    if len(vectors) != len(items):
        raise RuntimeError(
            f"Failed to build embeddings for all leaf nodes: expected {len(items)}, got {len(vectors)}"
        )

    for item, vector in zip(items, vectors):
        item['embedding'] = vector

    index = {'model': embedding_model, 'items': items}
    embedding_indexes[embedding_model] = index
    return index


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _collect_page_nums_from_ranges(items: list[dict]) -> list[int]:
    return sorted({
        p
        for item in items
        if item.get('start') is not None and item.get('end') is not None
        for p in range(item['start'], item['end'] + 1)
    })


def _get_content_for_page_nums(doc_info: dict, page_nums: list[int]) -> list[dict]:
    if not page_nums:
        return []
    if doc_info.get('type') == 'pdf':
        return _get_pdf_page_content(doc_info, page_nums)
    return _get_md_page_content(doc_info, page_nums)


def _rank_embedding_items(doc_info: dict, query: str, embedding_model: str, top_k: int) -> list[dict]:
    index = build_embedding_index(doc_info, embedding_model)
    items = index.get('items', [])
    if not items:
        return []

    query_embedding = embedding_completion(model=embedding_model, input_texts=[query])
    if not query_embedding:
        raise RuntimeError('Failed to create query embedding')
    query_vector = query_embedding[0]
    top_k = min(max(int(top_k or 5), 1), len(items))

    ranked_items = []
    for item in items:
        ranked_item = dict(item)
        ranked_item['score'] = _cosine_similarity(query_vector, item.get('embedding', []))
        ranked_items.append(ranked_item)
    ranked_items.sort(key=lambda item: item['score'], reverse=True)
    return ranked_items[:top_k]


def _select_hybrid_candidates(query: str, candidates: list[dict], model: str, top_k: int) -> list[dict]:
    if not candidates:
        return []
    candidate_payload = [
        {
            'index': i,
            'title': item.get('title', ''),
            'summary': item.get('summary', ''),
            'start': item.get('start'),
            'end': item.get('end'),
            'score': round(item.get('score', 0.0), 4),
        }
        for i, item in enumerate(candidates)
    ]
    prompt = (
        "You are reranking document sections for retrieval.\n\n"
        f"Query: {query}\n\n"
        f"Candidate sections:\n{json.dumps(candidate_payload, indent=2, ensure_ascii=False)}\n\n"
        f"Return a JSON array with up to {top_k} candidate indices that are most useful for answering the query. "
        "Prefer precise sections over loosely related ones. Return [] if none are relevant. "
        "Directly return the JSON array and nothing else."
    )

    response = llm_completion(model=model, prompt=prompt)
    try:
        indices = extract_json(response)
        if isinstance(indices, list):
            selected = []
            seen = set()
            for index in indices:
                if isinstance(index, int) and 0 <= index < len(candidates) and index not in seen:
                    selected.append(candidates[index])
                    seen.add(index)
                if len(selected) >= top_k:
                    break
            return selected
    except Exception:
        pass

    logging.warning(
        "search_document_hybrid: failed to parse rerank response; falling back to top embedding candidates."
    )
    return candidates[:top_k]


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

    Args:
        documents: Mapping of doc_id to document info dicts (as held by PageIndexClient).
        doc_id: Identifier of the document to search.
        query: Natural-language question or search query.
        model: LLM model name to use for node selection (defaults to the caller's model).

    Returns:
        JSON-encoded list of {'page': int, 'content': str} dicts for the relevant sections,
        or a JSON object with an 'error' key on failure.
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
        # Fallback: log a warning and treat all nodes as relevant so no content is lost
        logging.warning(
            "search_document: failed to parse LLM node-selection response; "
            "falling back to selecting all %d nodes at this level.",
            len(nodes),
        )
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


def search_document_by_embedding(documents: dict, doc_id: str, query: str, embedding_model: str, top_k: int = 5) -> str:
    """
    Search for relevant content using precomputed leaf-node embeddings.

    Returns the same JSON payload as search_document(): a list of {'page': int, 'content': str}.
    """
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({'error': f'Document {doc_id} not found'})
    if not embedding_model:
        return json.dumps({'error': 'Embedding search requires an embedding_model'})
    if not query or not query.strip():
        return json.dumps({'error': 'Query must not be empty'})

    try:
        ranked_items = _rank_embedding_items(doc_info, query, embedding_model, top_k=top_k)
    except Exception as e:
        return json.dumps({'error': f'Embedding search failed: {e}'})

    page_nums = _collect_page_nums_from_ranges(ranked_items)
    if not page_nums:
        return json.dumps([])

    try:
        content = _get_content_for_page_nums(doc_info, page_nums)
    except Exception as e:
        return json.dumps({'error': f'Failed to read page content: {e}'})

    return json.dumps(content, ensure_ascii=False)


def search_document_hybrid(
    documents: dict,
    doc_id: str,
    query: str,
    model: str = None,
    embedding_model: str = None,
    top_k: int = 5,
    candidate_k: int = 8,
) -> str:
    """
    Search using embedding recall followed by LLM reranking of the recalled leaves.

    Returns the same JSON payload as search_document(): a list of {'page': int, 'content': str}.
    """
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({'error': f'Document {doc_id} not found'})
    if not embedding_model:
        return json.dumps({'error': 'Hybrid search requires an embedding_model'})
    if not model:
        return json.dumps({'error': 'Hybrid search requires a rerank model'})
    if not query or not query.strip():
        return json.dumps({'error': 'Query must not be empty'})

    try:
        top_k = max(int(top_k or 5), 1)
        candidate_k = max(int(candidate_k or top_k), top_k)
        candidates = _rank_embedding_items(doc_info, query, embedding_model, top_k=candidate_k)
        if not candidates:
            return json.dumps([])
        selected_items = _select_hybrid_candidates(query, candidates, model=model, top_k=top_k)
        if not selected_items:
            selected_items = candidates[:top_k]
    except Exception as e:
        return json.dumps({'error': f'Hybrid search failed: {e}'})

    page_nums = _collect_page_nums_from_ranges(selected_items)
    if not page_nums:
        return json.dumps([])

    try:
        content = _get_content_for_page_nums(doc_info, page_nums)
    except Exception as e:
        return json.dumps({'error': f'Failed to read page content: {e}'})

    return json.dumps(content, ensure_ascii=False)
