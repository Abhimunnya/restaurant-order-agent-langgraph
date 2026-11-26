"""
Restaurant order management agent built with LangGraph.

Flow:
    user_input -> llm -> order_confirm -> llm -> cook -> llm -> serve -> llm -> END

The `llm` node is the "brain": every other node reports back to it by writing
`status`, and the llm reads the state (status + retry counters) to decide what
to say to the user and which node runs next.

uv add langgraph langchain-groq python-dotenv
"""

# Standard library stuff Python already comes with — no installs needed for these
import difflib  # lets us find "close enough" text matches, used for catching dish-name typos
import os  # lets us read environment variables (like secret API keys)
import random  # used to fake a coin-flip for simulating cook/serve success or failure
import sys  # lets us tweak low-level system settings, like terminal text encoding
from typing import Annotated, Literal, Optional, TypedDict
# Annotated = attach extra info to a type hint (used later to tell LangGraph how to merge messages)
# Literal = restrict a value to one of a fixed set of exact strings
# Optional = this value is allowed to be empty/None
# TypedDict = a dict that's expected to always have specific keys with specific types

from dotenv import load_dotenv  # loads secret keys from a .env file into the environment
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # proper "who said this" message objects
from langchain_core.runnables import RunnableConfig  # a config object LangGraph passes into nodes (used here for test overrides)
from langchain_groq import ChatGroq  # LangChain's standard wrapper around the Groq AI provider
from langgraph.graph import END, START, StateGraph  # the core building blocks for building our agent's flowchart
from langgraph.graph.message import add_messages  # a rule telling LangGraph to APPEND new messages instead of overwriting old ones
from pydantic import BaseModel, Field  # lets us define a strict "form" the AI's reply must fill in, with real validation

load_dotenv()  # actually reads the .env file and loads GROQ_API_KEY etc. into os.environ

# Windows console defaults to cp1252 and chokes on fancy quotes the model likes to emit
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # force terminal to handle any character safely instead of crashing

MODEL = "openai/gpt-oss-120b"  # which AI model we're going to talk to
# Create the actual connection to the AI. temperature=0 means "always be as consistent/predictable as possible", not creative
llm = ChatGroq(model=MODEL, api_key=os.getenv("GROQ_API_KEY"), temperature=0)

# ============================================================
# MENU  (dish -> quantity in stock)
# ============================================================

# Our "database" — just a plain dictionary: dish name -> how many we have in stock
MENU: dict[str, int] = {
    "pizza": 5,
    "burger": 3,
    "biryani": 10,
    "dosa": 2,
    "pasta": 0,  # on the menu but sold out
}

ORDER_RETRIES = 3  # customer gets 3 total chances to place a valid/available order
COOK_RETRIES = 2  # kitchen is allowed to fail and retry cooking up to 2 times
SERVE_RETRIES = 2  # serving is allowed to fail and retry up to 2 times

COOK_SUCCESS_PROB = 0.6  # simulated 60% chance cooking succeeds (this is fake/random, not a real kitchen)
SERVE_SUCCESS_PROB = 0.6  # simulated 60% chance serving succeeds


def find_menu_item(name: str) -> str | None:
    """Fuzzy-match a dish name to a MENU key. Returns the key, or None if nothing is close."""
    name = name.strip().lower()  # clean up spacing/capitalization so "Pizza " matches "pizza"
    if name in MENU:
        return name  # exact match found, done
    # "margherita pizza" -> "pizza", "biryani rice" -> "biryani"
    for key in MENU:  # loop through every dish name on the menu
        if key in name or name in key:  # check if one text is "contained inside" the other, either direction
            return key
    # "piza", "burgr" -> closest spelling
    # difflib looks for the closest-SPELLED menu word; cutoff=0.6 means "only if it's at least 60% similar"
    close = difflib.get_close_matches(name, MENU.keys(), n=1, cutoff=0.6)
    return close[0] if close else None  # return the best guess, or None if nothing was close enough


# ============================================================
# STATE
# ============================================================


class OrderDetails(TypedDict):
    # a little "form" describing one order: what dish, how many wanted, how many we actually have
    dish_name: str
    required_quantity: int
    available_quantity: int  # written by order_confirm; 0 if dish not on menu


# Values `status` can hold. Nodes write the left column, the llm reads it and
# writes the right column to say where the graph goes next.
#
#   written by nodes            written by llm (= routing decision)
#   ---------------------       ----------------------------------
#   new                         awaiting_user   -> user_input
#   confirmed / partial /       placed          -> order_confirm
#     unavailable               cook            -> cook
#   ready / cook_failed         serve           -> serve
#   complete / serve_failed     done / failed   -> END
# Literal[...] locks `status` down to ONLY ever being one of these exact words — nothing else is allowed
Status = Literal[
    "new",
    "awaiting_user",
    "placed",
    "confirmed",
    "partial",
    "unavailable",
    "cook",
    "ready",
    "cook_failed",
    "serve",
    "complete",
    "serve_failed",
    "done",
    "failed",
]


class State(TypedDict):
    # This is the "shared notebook" that flows through every step of the agent.
    # Annotated[list, add_messages] means: don't overwrite this list when updated, APPEND to it instead
    messages: Annotated[list, add_messages]
    order: OrderDetails  # the current order's details
    status: Status  # what's currently happening / what to do next
    order_retries: int  # order attempts left (3)
    cook_retries: int  # cook RE-runs left (2); first cook is free, every re-cook costs one, whoever caused it
    serve_retries: int  # serve RE-runs left (2); first serve is free
    final_result: str  # ends up holding "COMPLETED" or a "NOT COMPLETED: ..." explanation


def initial_state() -> State:
    # Builds the very first version of the notebook, before anything has happened yet
    return {
        "messages": [],  # no conversation yet
        "order": {"dish_name": "", "required_quantity": 0, "available_quantity": 0},  # empty order
        "status": "new",  # starting state
        "order_retries": ORDER_RETRIES,  # full retry budgets to start
        "cook_retries": COOK_RETRIES,
        "serve_retries": SERVE_RETRIES,
        "final_result": "",  # not finished yet
    }


# ============================================================
# LLM HELPERS
# ============================================================


class OrderIntent(BaseModel):
    """What the LLM extracts from a user message."""
    # This is the exact "form" we're forcing the AI to fill out instead of replying in free text.
    # BaseModel (Pydantic) actually checks at runtime that the AI's reply really matches this shape.

    # intent can ONLY be one of these three exact words — nothing else allowed
    intent: Literal["new_order", "accept_partial", "unrelated"] = Field(
        description=(
            "new_order: user is ordering a dish with a quantity. "
            "accept_partial: user agrees to take the smaller available quantity. "
            "unrelated: message has nothing to do with ordering food."
        )
        # NOTE: this description text isn't just a comment for us — the AI actually reads it
        # to decide what to put here, so it needs to stay clear and accurate.
    )
    # Optional[str] = this blank can be left empty (defaults to None, Python's "nothing" value)
    dish_name: Optional[str] = Field(default=None, description="Dish name, lowercase, singular")
    quantity: Optional[int] = Field(default=None, description="Quantity requested (default 1 if omitted)")


# The instructions given to the AI every time we ask it to classify what the customer typed.
# {menu} and {partial_offer} are placeholders filled in with real values before sending.
EXTRACT_PROMPT = """You are the order-taking assistant of a restaurant. You ONLY handle food orders.
Read the user's latest message and classify it.

- If they order a dish, return intent=new_order with dish_name (lowercase, singular) and quantity (1 if not stated).
- If they are agreeing to go ahead with the smaller available quantity that was offered to them
  (e.g. "yes", "ok go ahead", "fine, give me what you have"), return intent=accept_partial.
- If the message is not about ordering food (general questions, chit-chat, coding help...), return intent=unrelated.

Menu (dish -> in stock): {menu}
Was the user just offered a partial order? {partial_offer}
"""


def extract_order(state: State) -> OrderIntent:
    """Ask the LLM to pull dish + quantity (or intent) out of the latest user message."""
    # Figure out: were we just offering the customer a smaller quantity? The AI needs this
    # context to correctly understand what a plain "yes" would mean.
    partial_offer = "yes" if state["status"] == "awaiting_user" and state["order"]["available_quantity"] > 0 else "no"
    system = EXTRACT_PROMPT.format(menu=MENU, partial_offer=partial_offer)  # fill in the template with real values
    # Wrap the AI connection so it now returns a validated OrderIntent object instead of free text
    structured = llm.with_structured_output(OrderIntent)
    # Find the customer's MOST RECENT message: walk the conversation backwards (reversed),
    # and grab the first one that's actually from the human (not from the AI itself).
    # This whole "for ... if ..." with no brackets is a generator — it checks items one at a time
    # instead of building a whole new list in memory, and next() grabs the first match it finds.
    last_user = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
    return structured.invoke([SystemMessage(system), last_user])  # send it off, get back the filled-in form


def say(instruction: str, state: State) -> str:
    """Ask the LLM to phrase a short message to the customer given the current state."""
    # Build instructions telling the AI exactly what to say and giving it the current situation
    system = (
        "You are the friendly assistant of a restaurant's order system. "
        "Write ONE or TWO short sentences to the customer. No markdown.\n"
        f"Current order: {state['order']}\n"
        f"Retries left -> order: {state['order_retries']}, cook: {state['cook_retries']}, serve: {state['serve_retries']}\n"
        f"What to tell the customer: {instruction}"
    )
    # Ask the AI, grab just the text of its reply (.content), and trim any extra blank space (.strip())
    return llm.invoke([SystemMessage(system)]).content.strip()


# ============================================================
# NODES
# ============================================================


def user_input(state: State, config: RunnableConfig) -> dict:
    """Read the next order from the user. Tests can inject `input_fn` via config."""
    # Normally just uses Python's real input() function; but if a TEST has supplied a fake
    # version via config, use that instead — this is what makes automated testing possible
    read = config.get("configurable", {}).get("input_fn", input)
    text = read("\nYou: ")  # actually wait for (real or fake) input
    return {"messages": [HumanMessage(text)]}  # hand back just the new message; add_messages will append it


def llm_node(state: State) -> dict:
    """The brain. Reads status + counters, talks to the user, and decides the next status."""
    status = state["status"]  # what just happened / what situation are we in
    order = dict(state["order"])  # make a copy of the order so we can safely edit it

    # ---- waiting on the user: parse what they typed ----------------------
    if status in ("new", "awaiting_user"):
        parsed = extract_order(state)  # ask the AI to understand what the customer just said

        # Customer agreed to take the smaller available amount we offered them
        if parsed.intent == "accept_partial" and status == "awaiting_user" and order["available_quantity"] > 0:
            order["required_quantity"] = order["available_quantity"]  # update the order to the smaller number
            msg = say(f"Confirm we are going ahead with {order['required_quantity']} x {order['dish_name']} and sending it to the kitchen.", {**state, "order": order})
            # {**state, "order": order} = make a copy of state but swap in our updated order (dict unpacking)
            return {"messages": [AIMessage(msg)], "order": order, "status": "cook"}  # skip stock-check, go straight to cooking

        # Customer placed a brand-new order
        if parsed.intent == "new_order" and parsed.dish_name:
            order = {
                "dish_name": parsed.dish_name.strip().lower(),  # normalize spacing/capitalization
                "required_quantity": max(1, parsed.quantity or 1),  # never allow less than 1
                "available_quantity": 0,  # not checked yet — order_confirm will fill this in
            }
            return {"order": order, "status": "placed"}  # routes to order_confirm next

        # unrelated / could not extract -> costs one order attempt
        retries = state["order_retries"] - 1  # use up one attempt
        if retries <= 0:
            msg = say("Politely say you are a food-ordering assistant only, and since no valid order was placed after 3 attempts you are closing the session.", state)
            return {"messages": [AIMessage(msg)], "order_retries": 0, "status": "failed", "final_result": "NOT COMPLETED: no valid order after 3 attempts"}
        msg = say("Politely say you are an AI agent for food ordering only, not a general-purpose assistant, and ask them to place a food order (dish + quantity).", state)
        return {"messages": [AIMessage(msg)], "order_retries": retries, "status": "awaiting_user"}  # go back and wait again

    # ---- order_confirm reported back ------------------------------------
    if status == "confirmed":
        # Stock check passed fully — tell the customer and move on to cooking
        msg = say(f"Confirm the order of {order['required_quantity']} x {order['dish_name']} is available and is being sent to the kitchen.", state)
        return {"messages": [AIMessage(msg)], "status": "cook"}

    if status in ("partial", "unavailable"):
        # Stock check found a problem (not enough, or none at all) — this also costs an order attempt
        retries = state["order_retries"] - 1
        if retries <= 0:
            msg = say("Apologize: the order could not be fulfilled and all 3 order attempts are used up, so the session is closing.", state)
            return {"messages": [AIMessage(msg)], "order_retries": 0, "status": "failed", "final_result": "NOT COMPLETED: order attempts exhausted"}
        if status == "partial":
            instruction = (
                f"Only {order['available_quantity']} of the {order['required_quantity']} {order['dish_name']} requested are available. "
                f"Ask if they want to go ahead with {order['available_quantity']} or place a different order. Mention {retries} attempt(s) left."
            )
        else:  # unavailable
            instruction = f"'{order['dish_name']}' is not available at all. Ask them to place a different order. Mention the menu and {retries} attempt(s) left."
        msg = say(instruction, state)
        return {"messages": [AIMessage(msg)], "order_retries": retries, "status": "awaiting_user"}  # go back and wait for a new reply

    # ---- cook reported back ---------------------------------------------
    if status == "ready":
        # Cooking succeeded — tell the customer, move on to serving
        msg = say("Say the food is ready and is now being served.", state)
        return {"messages": [AIMessage(msg)], "status": "serve"}

    if status == "cook_failed":
        if state["cook_retries"] > 0:
            # Still have cook-retries left — spend one and try cooking again
            print(f"[llm] cook failed -> retry cook (cook retries left: {state['cook_retries'] - 1})")
            msg = say("Say there was a problem in the kitchen and the dish is being cooked again.", state)
            return {"messages": [AIMessage(msg)], "status": "cook", "cook_retries": state["cook_retries"] - 1}
        # Out of cook-retries — give up entirely
        msg = say("Apologize sincerely: the kitchen failed to prepare the dish even after retrying, so the order is cancelled.", state)
        return {"messages": [AIMessage(msg)], "status": "failed", "final_result": "NOT COMPLETED: cook failed"}

    # ---- serve reported back --------------------------------------------
    if status == "complete":
        # The happy ending — order fully succeeded
        msg = say("Tell the customer their order is complete and enjoy the meal.", state)
        return {"messages": [AIMessage(msg)], "status": "done", "final_result": "COMPLETED"}

    if status == "serve_failed":
        # a re-serve needs a re-cook first, so this spends one retry of each
        if state["serve_retries"] > 0 and state["cook_retries"] > 0:
            print(f"[llm] serve failed -> retry cook+serve (cook retries left: {state['cook_retries'] - 1}, serve retries left: {state['serve_retries'] - 1})")
            msg = say("Say serving went wrong, the dish is being cooked once more and will be served again.", state)
            return {"messages": [AIMessage(msg)], "status": "cook", "cook_retries": state["cook_retries"] - 1, "serve_retries": state["serve_retries"] - 1}
        # Out of retries on one or both — give up, with a specific reason why
        reason = "serve failed" if state["serve_retries"] <= 0 else "serve failed and cook retries exhausted"
        msg = say(f"Apologize sincerely: {reason}, so the order is cancelled.", state)
        return {"messages": [AIMessage(msg)], "status": "failed", "final_result": f"NOT COMPLETED: {reason}"}

    # Safety net: if `status` was ever something none of the above checks handled, stop loudly
    # instead of silently doing nothing — this should never actually happen if the rest is correct
    raise ValueError(f"llm node got unexpected status {status!r}")


def order_confirm(state: State) -> dict:
    """Check the order against the MENU and write available_quantity + status."""
    order = dict(state["order"])  # copy so we can edit safely
    match = find_menu_item(order["dish_name"])  # try to fuzzy-match the dish against the real menu
    if match:
        order["dish_name"] = match  # correct the name to the official menu spelling
    available = MENU[match] if match else 0  # how many we actually have (0 if no match at all)
    order["available_quantity"] = available

    if available == 0:
        status = "unavailable"  # not on the menu / sold out
    elif available < order["required_quantity"]:
        status = "partial"  # some available, just not enough
    else:
        status = "confirmed"  # fully available

    print(f"[order_confirm] {order['dish_name']}: need {order['required_quantity']}, have {available} -> {status}")  # debug visibility
    return {"order": order, "status": status}


def _outcome(config: RunnableConfig, key: str, prob: float) -> bool:
    """60/40 coin flip. Tests can pre-script results with config['configurable'][key] = [True, False, ...]."""
    scripted = config.get("configurable", {}).get(key)  # check if a test has pre-decided the outcome
    if scripted:
        return bool(scripted.pop(0))  # use (and remove) the next pre-scripted result instead of rolling randomly
    return random.random() < prob  # real coin flip: random number under `prob` counts as success


def cook(state: State, config: RunnableConfig) -> dict:
    """60/40 coin flip. The llm decides (and pays for) any retry."""
    # This function ONLY reports success/failure — it never touches the retry counters itself;
    # the llm node is the one that decides whether to retry and spends the retry budget
    if _outcome(config, "cook_outcomes", COOK_SUCCESS_PROB):
        print("[cook] success -> ready")
        return {"status": "ready"}
    print("[cook] FAILED")
    return {"status": "cook_failed"}


def serve(state: State, config: RunnableConfig) -> dict:
    """60/40 coin flip. The llm decides (and pays for) any retry."""
    if _outcome(config, "serve_outcomes", SERVE_SUCCESS_PROB):
        print("[serve] success -> complete")
        return {"status": "complete"}
    print("[serve] FAILED")
    return {"status": "serve_failed"}


# ============================================================
# EDGES
# ============================================================


def route_after_llm(state: State) -> str:
    # Looks at whatever status the llm node JUST decided, and translates it into
    # "which node should actually run next" — this is the function LangGraph calls
    # on the conditional edge coming out of the llm node
    status = state["status"]
    if status == "awaiting_user":
        return "user_input"
    if status == "placed":
        return "order_confirm"
    if status in ("cook", "serve"):
        return status  # these two happen to share their status name with their node name
    return END  # covers "done" and "failed" — nothing left to route to, stop the graph


# ============================================================
# GRAPH
# ============================================================

builder = StateGraph(State)  # start building the flowchart, telling it our State shape upfront
# Register every node: a string name, plus the real function to run for that step
builder.add_node("user_input", user_input)
builder.add_node("llm", llm_node)
builder.add_node("order_confirm", order_confirm)
builder.add_node("cook", cook)
builder.add_node("serve", serve)

builder.add_edge(START, "user_input")  # graph always starts by asking the user something
# Every worker node unconditionally reports back to the "llm" brain node — fixed, no branching here
builder.add_edge("user_input", "llm")
builder.add_edge("order_confirm", "llm")
builder.add_edge("cook", "llm")
builder.add_edge("serve", "llm")
# The ONLY branching point in the whole graph: after "llm" runs, ask route_after_llm
# which path to take next, using this dict to map its answer to a real node name
builder.add_conditional_edges(
    "llm",
    route_after_llm,
    {"user_input": "user_input", "order_confirm": "order_confirm", "cook": "cook", "serve": "serve", END: END},
)

graph = builder.compile()  # finalize the flowchart into something actually runnable


# ============================================================
# RUN
# ============================================================


def run(config: RunnableConfig | None = None) -> State:
    """Run one ordering session. Prints every AI message as it is produced."""
    # recursion_limit caps how many steps the graph can take total, as a safety net against infinite loops
    config = {"recursion_limit": 100, **(config or {})}
    seen = 0  # how many messages we've already printed, so we don't print the same ones twice
    final = None
    # stream() (instead of invoke()) hands us the state AFTER EVERY node, not just at the very end,
    # so we can print new AI messages live as they're produced
    for final in graph.stream(initial_state(), config=config, stream_mode="values"):
        for m in final["messages"][seen:]:  # only look at messages we haven't printed yet
            if isinstance(m, AIMessage):  # only print the AI's own replies, not the customer's
                print(f"Agent: {m.content}")
        seen = len(final["messages"])  # update how many we've now printed
    print(f"\nFINAL RESULT: {final['final_result']}")
    return final


def simulate(user_inputs: list[str], cook_outcomes: list[bool] | None = None, serve_outcomes: list[bool] | None = None) -> State:
    """Run a scripted scenario: canned user replies and pre-decided cook/serve results."""
    inputs = list(user_inputs)  # make our own copy of the list of fake replies to use up

    def fake_input(prompt: str) -> str:
        # Stand-in for real input(): returns the next scripted reply instead of waiting on a real person
        text = inputs.pop(0) if inputs else "bye"  # if we've run out of scripted replies, default to "bye"
        print(f"{prompt}{text}")
        return text

    # Bundle up all our test overrides and run the real graph with them injected via config
    return run({"configurable": {"input_fn": fake_input, "cook_outcomes": list(cook_outcomes or []), "serve_outcomes": list(serve_outcomes or [])}})


if __name__ == "__main__":
    # Only runs when this file is executed directly (not when imported elsewhere)
    print(f"Welcome! Menu: {MENU}")
    print("Tell me what you'd like to order (one dish + quantity).")
    run()  # kick off a real session with a real human typing at the keyboard