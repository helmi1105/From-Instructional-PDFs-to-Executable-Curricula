"""Decode declared page encodings without guessing from array length."""


def decode_pages(pages, semantics=None):
    if not isinstance(pages, list):
        raise ValueError('pages must be a list')
    if any(type(p) is not int or p < 1 for p in pages):
        raise ValueError('pages must contain positive integer PDF page numbers')
    semantics = semantics or {}
    fmt = semantics.get('format')
    if fmt is None:
        return sorted(set(pages))
    if fmt.replace(' ', '') != '[start_page,end_page]' or semantics.get('inclusive') is not True:
        raise ValueError(f'Unsupported page semantics: {semantics!r}')
    if not pages:
        return []
    if len(pages) != 2 or pages[0] > pages[1]:
        raise ValueError('Inclusive page interval requires [start, end] with start <= end')
    return list(range(pages[0], pages[1] + 1))


def gold_pages_by_id(data):
    """Prefer grounding.pages, falling back to node.pages when absent.

    An explicit empty grounding list remains empty. Undeclared lists retain
    enumeration semantics, including two-element lists in historical references.
    """
    semantics = data.get('page_semantics')
    grounding = data.get('grounding', {}) or {}
    result = {}
    for node in data.get('nodes', []):
        entry = grounding.get(node['id'], {}) or {}
        pages = entry['pages'] if 'pages' in entry else node.get('pages', [])
        try:
            result[node['id']] = decode_pages(pages, semantics)
        except ValueError as exc:
            raise ValueError(f"Node {node['id']}: {exc}") from exc
    return result
