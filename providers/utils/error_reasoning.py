import logging

from core.chat_history import ChatHistoryMessage, ChatHistoryController
from providers.base import BaseLLM


async def error_reasoning(
        error_message: str,
        llm: BaseLLM,
        chat: ChatHistoryController,
):

    instructions = chat.history[0].content
    assistant_messages = []
    user_message = ""

    for message in reversed(chat.history):

        logging.info(message)

        if message.role == "user":
            logging.info("ist user message -> break")
            user_message = message.content if message.content else ""
            break

        assistant_messages.insert(0, message.content if message.content else "")

    # Formatierung

    context = f"""
***DEINE AUFGABE***
Du hilfst einem KI Assistenten einen Fehler zu beheben.
Als Überblick bekommst du:
 - die Instruktionen, die dieser Assistent bekommen hat
 - die letzten relevanten Nachrichten
 - die Fehlermeldung

Erwähne zuerst einmal welcher Fehler aufgetreten ist.
Erkläre dann klar und möglichst knapp wie der Fehler entstanden ist und wie er behoben werden kann.


***Instruktionen für den Assistenten***

\"{instructions}\"


***Letzte Nachricht des Nutzers***

\"{user_message}\"


***Darauffolgende Nachrichten des Assistenten***

\"{"\n---\n".join(assistant_messages)}\"


***Die Fehlermeldung***

\"{error_message}\"
"""

    logging.info(context)

    reasoning_chat = await llm.get_empty_history_controller()
    reasoning_chat.history.append(ChatHistoryMessage(role="system", content=context))

    reasoning = await llm.generate(reasoning_chat)

    reasoning_content = reasoning.text

    logging.info(reasoning_content)

    return reasoning_content