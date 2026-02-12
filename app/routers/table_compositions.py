"""
Table compositions router
Endpoints for table-centric recovered workflows and sequential recommendations
"""

from typing import List
from fastapi import APIRouter, Depends, Query, Body
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services import table_compositions_service

router = APIRouter()


@router.get("/")
async def list_table_compositions(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """
    List TableCompositions stored in DB.

    These are produced by GET /compositions/recoverNew (recover_new),
    which persists both Compositions and TableCompositions.
    """
    return await table_compositions_service.list_table_compositions(db=db, limit=limit, offset=offset)


@router.post("/predict")
async def predict_next(
    table_sequence: List[int] = Body(..., description="Current sequence of table/dataset IDs in the workflow"),
    n: int = Body(5, ge=1, le=20, description="Number of predictions to return"),
    db: AsyncSession = Depends(get_db),
):
    """
    Predict the next table in a workflow sequence.

    Uses the DAGNN (graph neural network) model trained on compositionsDAG.json.
    The model must be trained first via POST /sequential/train.

    Input: table_sequence — array of table IDs already in the workflow.
    Output: predictions ranked by DAGNN score + confidence.

    Example:
    ```json
    {
        "table_sequence": [1003093, 1003086],
        "n": 5
    }
    ```
    """
    return await table_compositions_service.predict_next(
        db=db,
        table_sequence=table_sequence,
        n=n,
    )
