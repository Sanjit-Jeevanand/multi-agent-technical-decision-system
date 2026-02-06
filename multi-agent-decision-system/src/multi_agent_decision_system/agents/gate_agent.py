from multi_agent_decision_system.core.schema import GateOutput, GateDecision

def run_gate(state) -> GateOutput:
    """
    Deterministic gate check.
    No LLM call.
    """

    critic = state.current.critic
    synthesizer = state.current.synthesizer

    blocking_reasons = []
    required_actions = []

    if critic and critic.requires_revision:
        blocking_reasons.append("Critic requires revision due to high-impact issues.")
        required_actions.append("Address high-impact critic issues before proceeding.")

    if synthesizer.final_recommendation == "insufficient_information":
        blocking_reasons.append("Synthesizer reports insufficient information.")
        required_actions.append("Provide missing inputs or clarify constraints.")

    approved = len(blocking_reasons) == 0

    return GateOutput(
        agent_name="gate",
        decision=GateDecision(
            approved=approved,
            blocking_reasons=blocking_reasons,
            required_actions=required_actions,
        ),
    )