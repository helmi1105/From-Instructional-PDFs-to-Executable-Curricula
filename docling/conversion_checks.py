"""Validate actual processed-page evidence, never placeholder document page records."""


def conversion_complete(status, errors, processed_pages, expected_pages):
    return (status == 'success' and not errors
            and set(processed_pages) == set(expected_pages)
            and len(processed_pages) == len(set(expected_pages)))
