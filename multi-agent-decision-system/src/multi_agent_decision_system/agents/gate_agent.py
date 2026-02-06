from multi_agent_decision_system.core.schema import GateOutput, GateDecision
from multi_agent_decision_system.core.state_v2 import DecisionState


def run_gate(state: DecisionState) -> GateOutput:
    critic = state.current.critic
    synthesizer = state.current.synthesizer

    # Escape hatch: explicit force approve
    if getattr(state, "force_approve", False):
        return GateOutput(
            agent_name="gate",
            decision=GateDecision(
                approved=True,
                blocking_reasons=[],
                required_actions=[],
            ),
        )

    # Missing decision
    if synthesizer is None or synthesizer.final_recommendation is None:
        return GateOutput(
            agent_name="gate",
            decision=GateDecision(
                approved=False,
                blocking_reasons=["No final recommendation produced."],
                required_actions=["Rerun decision or inspect synthesizer output."],
            ),
        )

    # Approval rule: approve only if critic does NOT require revision
    if critic and not critic.requires_revision:
        return GateOutput(
            agent_name="gate",
            decision=GateDecision(
                approved=True,
                blocking_reasons=[],
                required_actions=[],
            ),
        )

    # Block otherwise and require explicit user action
    return GateOutput(
        agent_name="gate",
        decision=GateDecision(
            approved=False,
            blocking_reasons=[
                "Critic requires revision; explicit user action is required."
            ],
            required_actions=[
                "Accept flagged risks or update constraints to proceed."
            ],
        ),
    )