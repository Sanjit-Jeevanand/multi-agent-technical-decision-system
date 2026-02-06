from multi_agent_decision_system.core.schema import GateOutput, GateDecision
from multi_agent_decision_system.core.state_v2 import DecisionState, GatePolicyTier


def run_gate(state: DecisionState) -> GateOutput:
    tier = state.gate_tier
    critic = state.current.critic
    synthesizer = state.current.synthesizer

    if tier == GatePolicyTier.OVERRIDE:
        return GateOutput(
            agent_name="gate",
            decision=GateDecision(
                approved=True,
                blocking_reasons=[],
                required_actions=[],
            ),
        )

    if synthesizer is None or synthesizer.final_recommendation is None:
        return GateOutput(
            agent_name="gate",
            decision=GateDecision(
                approved=False,
                blocking_reasons=["No final recommendation produced."],
                required_actions=["Rerun decision or inspect synthesizer output."],
            ),
        )

    if tier == GatePolicyTier.EXPLORATION:
        if critic and critic.requires_revision:
            return GateOutput(
                agent_name="gate",
                decision=GateDecision(
                    approved=False,
                    blocking_reasons=["Critic requires revision."],
                    required_actions=["Revise based on critic feedback."],
                ),
            )
        else:
            return GateOutput(
                agent_name="gate",
                decision=GateDecision(
                    approved=True,
                    blocking_reasons=[],
                    required_actions=[],
                ),
            )

    if tier == GatePolicyTier.COMMITMENT:
        return GateOutput(
            agent_name="gate",
            decision=GateDecision(
                approved=True,
                blocking_reasons=[],
                required_actions=[],
            ),
        )

    # Default fallback: approve
    return GateOutput(
        agent_name="gate",
        decision=GateDecision(
            approved=True,
            blocking_reasons=[],
            required_actions=[],
        ),
    )