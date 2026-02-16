"""
Table compositions service.

Provides:
- listing table compositions stored in DB
- predict: delegates to existing SequentialDAGNN model (predict_next_table)

No frequency-counting — uses the trained DAGNN graph neural network.
"""

from __future__ import annotations

from typing import Any, Dict, List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import TableComposition
from app.services import sequential_recommendations_service


async def list_table_compositions(db: AsyncSession, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
    q = select(TableComposition).offset(offset).limit(limit)
    result = await db.execute(q)
    comps = result.scalars().all()
    return {
        "items": [
            {
                "id": c.id,
                "table_ids": c.table_ids,
            }
            for c in comps
        ],
        "limit": limit,
        "offset": offset,
        "returned": len(comps),
    }


async def predict_next(
    db: AsyncSession,
    table_sequence: List[int],
    n: int = 5,
) -> Dict[str, Any]:
    """
    Predict next table in a workflow sequence using DAGNN model.

    Delegates to the existing SequentialDAGNNAlgorithm (trained on compositionsDAG.json).
    This ensures predictions use the graph neural network, not simple frequency counting.

    Input:  table_sequence = [1003093, 1003086]
    Output: DAGNN-based predictions with scores and confidence
    """
    return await sequential_recommendations_service.predict_next_table(
        table_sequence=table_sequence,
        n=n,
        db=db,
    )
