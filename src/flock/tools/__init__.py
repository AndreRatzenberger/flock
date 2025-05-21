from .azure_tools import (
    azure_search_create_index,
    azure_search_create_vector_index,
    azure_search_delete_documents,
    azure_search_get_document,
    azure_search_get_index_statistics,
    azure_search_initialize_clients,
    azure_search_list_indexes,
    azure_search_query,
    azure_search_upload_documents,
    azure_storage_create_container,
    azure_storage_delete_blob,
    azure_storage_delete_container,
    azure_storage_download_blob_to_bytes,
    azure_storage_download_blob_to_file,
    azure_storage_download_blob_to_text,
    azure_storage_get_blob_properties,
    azure_storage_list_blobs,
    azure_storage_list_containers,
    azure_storage_upload_blob_bytes,
    azure_storage_upload_blob_from_file,
    azure_storage_upload_blob_text,
)
from .code_tools import code_code_eval, code_evaluate_math
from .file_tools import (
    file_get_anything_as_markdown,
    file_json_parse_safe,
    file_json_search,
    file_read_from_file,
    file_save_to_file,
)
from .github_tools import (
    github_create_files,
    github_create_user_stories_as_github_issue,
    github_upload_readme,
)
from .markdown_tools import (
    markdown_extract_code_blocks,
    markdown_extract_links,
    markdown_extract_tables,
    markdown_split_by_headers,
    markdown_to_plain_text,
)
from .text_tools import (
    text_calculate_hash,
    text_chunking_for_embedding,
    text_clean_text,
    text_count_tokens,
    text_count_tokens_estimate,
    text_count_words,
    text_detect_language,
    text_extract_json_from_text,
    text_extract_keywords,
    text_extract_numbers,
    text_extract_urls,
    text_format_chat_history,
    text_format_table_from_dicts,
    text_recursive_splitter,
    text_split_by_characters,
    text_split_by_sentences,
    text_split_by_separator,
    text_split_by_tokens,
    text_split_code_by_functions,
    text_tiktoken_split,
    text_truncate_to_token_limit,
)
from .web_tools import (
    web_content_as_markdown,
    web_search_bing,
    web_search_duckduckgo,
    web_search_tavily,
)
from .zendesk_tools import (
    zendesk_get_article_by_id,
    zendesk_get_articles,
    zendesk_get_comments_by_ticket_id,
    zendesk_get_ticket_by_id,
    zendesk_get_tickets,
    zendesk_search_articles,
)

storage_tools = [
    azure_storage_list_containers,
    azure_storage_create_container,
    azure_storage_delete_container,
    azure_storage_list_blobs,
    azure_storage_upload_blob_text,
    azure_storage_upload_blob_bytes,
    azure_storage_upload_blob_from_file,
    azure_storage_download_blob_to_text,
    azure_storage_download_blob_to_bytes,
    azure_storage_download_blob_to_file,
    azure_storage_delete_blob,
    azure_storage_get_blob_properties,
]

azure_search_tools = [
    azure_search_initialize_clients,
    azure_search_create_index,
    azure_search_upload_documents,
    azure_search_query,
    azure_search_get_document,
    azure_search_delete_documents,
    azure_search_list_indexes,
    azure_search_get_index_statistics,
    azure_search_create_vector_index,
]

file_tools_collection = [
    file_get_anything_as_markdown,
    file_save_to_file,
    file_read_from_file,
    file_json_parse_safe,
    file_json_search,
]

code_tools_collection = [code_evaluate_math, code_code_eval]

web_tools_collection = [
    web_content_as_markdown,
    web_search_bing,
    web_search_duckduckgo,
    web_search_tavily,
]

github_tools_collection = [
    github_create_user_stories_as_github_issue,
    github_upload_readme,
    github_create_files,
]

llm_processing_tools = [
    text_split_by_sentences,
    text_split_by_characters,
    text_split_by_tokens,
    text_split_by_separator,
    text_recursive_splitter,
    text_chunking_for_embedding,
    text_split_code_by_functions,
    text_count_tokens,
    text_count_tokens_estimate,
    text_truncate_to_token_limit,
    text_extract_keywords,
    text_clean_text,
    text_format_chat_history,
    text_extract_json_from_text,
    text_calculate_hash,
    text_format_table_from_dicts,
    text_detect_language,
    text_tiktoken_split,
    text_count_words,
    text_extract_urls,
    text_extract_numbers,
]

markdown_processing_tools = [
    markdown_split_by_headers,
    markdown_extract_code_blocks,
    markdown_extract_links,
    markdown_extract_tables,
    markdown_to_plain_text,
]

zendesk_tools_collection = [
    zendesk_get_tickets,
    zendesk_get_ticket_by_id,
    zendesk_get_comments_by_ticket_id,
    zendesk_get_article_by_id,
    zendesk_get_articles,
    zendesk_search_articles,
]

__all__ = [
    "azure_search_tools",
    "code_tools_collection",
    "file_tools_collection",
    "github_tools_collection",
    "llm_processing_tools",
    "markdown_processing_tools",
    "storage_tools",
    "web_tools_collection",
    "zendesk_tools_collection",
]

# If there was existing content in __init__.py, this approach might overwrite it.
# A safer approach if __init__.py might exist and have other critical initializations
# would be to read it first, then append/modify.
# For now, assuming a fresh creation or simple __init__.py.
