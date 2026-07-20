from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from auth.authenticate import authenticate_cookie
from case_browser import get_case_detail, slice_to_png_bytes

cases_route = APIRouter()


@cases_route.get(
    "/{patient_id}",
    summary="Карточка пациента из базы данных",
)
async def get_historical_case(
    patient_id: str,
    _user_email: str = Depends(authenticate_cookie),
):
    try:
        return get_case_detail(patient_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@cases_route.get(
    "/{patient_id}/slices/{filename}",
    summary="Срез исторического МРТ пациента",
)
async def get_historical_slice(
    patient_id: str,
    filename: str,
    with_mask: bool = Query(False),
    _user_email: str = Depends(authenticate_cookie),
):
    try:
        png = slice_to_png_bytes(patient_id, filename, with_mask=with_mask)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(content=png, media_type="image/png")
