"""Browser previews and authenticated downloads over explicit workspace roots."""

from fastapi import APIRouter, HTTPException, Query
from starlette.responses import FileResponse, Response

from ..services.artifacts import ArtifactService, read_text_preview


def _http_error(error: Exception) -> HTTPException:
    if isinstance(error, PermissionError):
        return HTTPException(403, str(error))
    if isinstance(error, FileNotFoundError):
        return HTTPException(404, str(error))
    return HTTPException(400, str(error))


def file_router(artifacts: ArtifactService) -> APIRouter:
    router = APIRouter(prefix="/api/agent/v1/file")

    @router.get("")
    def preview(
        path: str = Query(...), workspace_id: str = Query(default="w_main")
    ) -> Response:
        try:
            resolved, kind = artifacts.resolve_preview(workspace_id, path)
            if kind == "image":
                return FileResponse(resolved)
            if kind == "binary":
                return FileResponse(
                    resolved,
                    filename=resolved.name,
                    media_type="application/octet-stream",
                )
            text, truncated = read_text_preview(resolved)
            return Response(
                content=text,
                media_type="text/plain; charset=utf-8",
                headers={
                    "X-File-Truncated": "1" if truncated else "0",
                    "X-File-Total-Bytes": str(resolved.stat().st_size),
                },
            )
        except (ValueError, FileNotFoundError, PermissionError) as error:
            raise _http_error(error) from None

    @router.get("/meta")
    def metadata(path: str = Query(...), workspace_id: str = Query(default="w_main")):
        try:
            return artifacts.artifact_meta(workspace_id, path)
        except (ValueError, FileNotFoundError, PermissionError) as error:
            raise _http_error(error) from None

    @router.get("/list")
    def list_files(
        path: str | None = Query(default=None),
        workspace_id: str = Query(default="w_main"),
        limit: int = Query(default=2000, ge=1, le=20000),
    ):
        try:
            return artifacts.list_artifacts(
                workspace_id, subdir=path, limit=limit, recursive=False, preview=True
            )
        except (ValueError, FileNotFoundError, PermissionError) as error:
            raise _http_error(error) from None

    return router


def artifact_router(artifacts: ArtifactService) -> APIRouter:
    """Include this router beneath the tools token dependency."""
    router = APIRouter(prefix="/workspaces/{workspace_id}/artifacts")

    @router.get("")
    def list_artifacts(
        workspace_id: str,
        subdir: str | None = Query(default=None),
        limit: int = Query(default=2000, ge=1, le=20000),
    ):
        try:
            return artifacts.list_artifacts(workspace_id, subdir=subdir, limit=limit)
        except (ValueError, FileNotFoundError, PermissionError) as error:
            raise _http_error(error) from None

    @router.get("/download")
    def download(workspace_id: str, path: str = Query(...)) -> FileResponse:
        try:
            resolved = artifacts.resolve_artifact(workspace_id, path)
        except (ValueError, FileNotFoundError, PermissionError) as error:
            raise _http_error(error) from None
        return FileResponse(resolved, filename=resolved.name)

    return router
