"""Represents a Resource received from a MCP Server"""


from typing import Annotated, Literal, TypeVar
from pydantic import AnyUrl, BaseModel, Field, UrlConstraints

from mcp.types import Resource, ReadResourceResult, TextResourceContents, BlobResourceContents

from flock.core.logging.logging import get_logger


logger = get_logger("core.mcp.resource_base")
T = TypeVar("T", bound="FlockMCPResourceBase")


class FlockMCPResourceBase(BaseModel):
    """
    Documentation:
        https://modelcontextprotocol.io/docs/concepts/resources
    Summary:
        Resources represent any kind of data that an MCP Server
        wants to make available to clients. This can include:
        - File contents
        - Database records
        - API responses
        - Live system data
        - Screenshots and images
        - Log files
        - and more

        Each resource is identified by a unique URI and can contain either
        text or binary data (base64 encoded)
    """
    data_type: Literal["text", "base64"] | None = Field(
        ...,
        description="Resources can contain two types of content: UTF-8 encoded text data OR Binary data encoded in base64."
    )

    name: str = Field(
        ...,
        description="Unique identifier for the resource."
    )

    description: str | None = Field(
        default=None,
        description="Human-readable name."
    )

    uri: Annotated[AnyUrl, UrlConstraints(host_required=False)] = Field(
        ...,
        description="Resources are identified using URIs that follow this format: [protocol]://[host]/[path]"
    )

    mime_type: str | None = Field(
        default=None,
        description="(Optional) Mime-Type of the Resource."
    )

    size: int | None = Field(
        default=None,
        description="The size of the raw resource content, in bytes before base64 encoding, if known."
    )

    data: list[str] | None = Field(
        default=None,
        description="Can be a string of text or a string of base64-encoded binary data."
    )

    @classmethod
    def try_to_mcp_resource(cls: type[T], resource: T) -> tuple[Resource | None, ReadResourceResult | None] | None:
        """Attempt to convert a flock mcp Resource into a mcp resource."""
        try:
            if resource.uri:
                logger.debug(f"Converting {resource.name} to mcp resource.")

                data: list[str] | None = None

                if hasattr(resource, "data") and resource.data:
                    data = resource.data

                if hasattr(resource, "data_type") and resource.data_type and resource.data_type == "base64" or resource.data_type == "text":
                    data_result = None
                    resource_result = Resource(
                        uri=resource.uri,
                        name=resource.name,
                        description=resource.description,
                        mimeType=resource.mime_type,
                        size=resource.size,
                    )
                    if resource.data_type == "base64":

                        data_contents = [BlobResourceContents(
                            uri=resource.uri, mimeType=resource.mime_type, blob=b) for b in data]

                        data_result = ReadResourceResult(
                            contents=data_contents
                        )

                    if resource.data_type == "text":

                        data_contents = [TextResourceContents(
                            uri=resource.uri, mimeType=resource.mime_type, text=t) for t in data]

                    return (resource_result, data_result)

                else:
                    logger.warning(
                        f"Resource does not declare its data_type. Skipping...")

            else:
                logger.warning(f"Resource does not contain a uri. Skipping...")
                return None

        except Exception as e:
            logger.error(
                f"Excpetion when attempting to convert flock resource {resource.name} to mcp resource: {e}")
            return None

    @classmethod
    def try_from_mcp_resource(cls: type[T], resource: Resource, contents: ReadResourceResult | None = None) -> type[T] | None:
        """Attempt to convert a mcp Resource to a flock resource. """
        try:
            if resource.uri:
                logger.debug(
                    f"Converting {resource.name} to flock mcp_resource.")

                data = []
                type = None
                if contents:
                    logger.debug(
                        f"Resource {resource.name} contains content. Converting...")
                    data_contents = contents.contents

                    if isinstance(data_contents, list[TextResourceContents]):
                        type = "text"
                        for text_content in data_contents:
                            if hasattr(text_content, "text") and text_content.text:
                                data.append(text_content.text)

                    if isinstance(data_contents, list[BlobResourceContents]):
                        type = "base64"
                        for blob_content in data_contents:
                            if hasattr(blob_content, "blob") and blob_content.blob:
                                data.append(blob_content.blob)
                    else:
                        data = None

                result = cls(
                    name=resource.name if resource.name else "unknown_resource",
                    description=resource.description,
                    uri=resource.uri,
                    mime_type=resource.mimeType,
                    size=resource.size,
                    data_type=type,
                    data=data,
                )
            else:
                return None
        except Exception as e:
            logger.error(
                f"Exception when attempting to convert resource: {resource.name} to flock resource: {e}")
            return None
