"""Default webhook agent."""

triggers = {"webhook": True}


async def default(context):
    agent = await context.init()
    session = await agent.session(context.agent_id)
    result = await session.prompt(context.payload["prompt"])
    return {"text": result.text}
