
import asyncio
from fastapi import APIRouter, HTTPException
from mysite.chain.ticket import ticket_chain
from mysite.database.schemas import TicketInputSchema, TicketOutputSchema, AnswerInputSchema, AnswerOutputSchema
from langchain_core.exceptions import OutputParserException
from mysite.chain.answer_chain import answer_chain

ai_router = APIRouter(prefix='/ai', tags=['AI Assistant'])


@ai_router.post('/analyze', response_model=TicketOutputSchema)
async def analyze_ticke(ticket: TicketInputSchema):
    text = ticket.text.strip()

    if not text:
        raise HTTPException(status_code=422, detail='Маалымат туура эмес форматта берилген')

    try:
        result = await asyncio.wait_for(
            ticket_chain.ainvoke({'text': text}), timeout=120
        )
    except TimeoutError:
        raise HTTPException(status_code=500, detail='Модель тапшырманы аткарганы текшерген жок')
    except OutputParserException:
        raise HTTPException(status_code=422, detail='Моделдин жообу схемага туура келбейт')

    return result


@ai_router.post('/answer', response_model=AnswerOutputSchema)
async def answer_list(answer: AnswerInputSchema):
    text = answer.text.strip()
    facts = answer.facts.strip()

    if not text or not facts:
        raise HTTPException(status_code=422, detail='Маалымат туура эмес форматта берилген')

    try:
        result = await asyncio.wait_for(
            answer_chain.ainvoke({'text': text, 'facts': facts}), timeout=120
        )

    except TimeoutError:
        raise HTTPException(status_code=500, detail='Модель тапшырманы аткарганы текшерген жок')
    except OutputParserException:
        raise HTTPException(status_code=422, detail='Моделдин жообу схемага туура келбейт')

    result = result.strip()
    return AnswerOutputSchema(text_draft=result)
