"""
Generate a real debate transcript using LLM for PR documentation.
"""

from loft.dialectical.debate_framework import DebateFramework
from loft.dialectical.debate_schemas import DebateContext
from loft.dialectical.critic import CriticSystem
from loft.dialectical.synthesizer import Synthesizer
from loft.neural.rule_generator import RuleGenerator
from loft.neural.llm_interface import LLMInterface
from loft.neural.providers import AnthropicProvider
from loft.symbolic.asp_core import ASPCore
import os
from dotenv import load_dotenv

# Load .env configuration
load_dotenv()

# Get configuration from environment
api_key = os.environ.get("ANTHROPIC_API_KEY")
model = os.environ.get("LLM_MODEL", "claude-3-5-haiku-20241022")

if not api_key:
    print("Error: ANTHROPIC_API_KEY not found in .env file")
    print("Please add it to .env: ANTHROPIC_API_KEY=your-key-here")
    exit(1)

# Initialize Anthropic provider
print(f"Initializing Anthropic provider with model: {model}")
provider = AnthropicProvider(
    api_key=api_key,
    model=model,
    max_tokens=4096,
)

llm = LLMInterface(provider=provider)

# Initialize ASP core
print("Initializing ASP core...")
asp_core = ASPCore()

# Initialize agents with real LLM
print("Initializing debate agents...")
generator = RuleGenerator(llm=llm, asp_core=asp_core, domain="contract_law")
critic = CriticSystem(llm_client=llm, mock_mode=False)
synthesizer = Synthesizer(llm_client=llm, mock_mode=False)

# Create debate framework
print("Creating debate framework...")
framework = DebateFramework(
    generator=generator,
    critic=critic,
    synthesizer=synthesizer,
    max_rounds=3,
    convergence_threshold=0.85
)

# Define debate context
print("\nStarting dialectical debate...")
print("=" * 80)
context = DebateContext(
    knowledge_gap_description="A contract is enforceable when there is mutual agreement, consideration exchanged, legal capacity of parties, and lawful purpose",
    existing_rules=[],
    existing_predicates=["enforceable", "contract", "signed", "offer", "acceptance", "consideration", "capacity", "legal_purpose"],
    target_layer="tactical",
    max_rounds=3,
)

# Run the debate
result = framework.run_dialectical_cycle(context)

# Print results
print("\n" + "=" * 80)
print("DEBATE RESULTS")
print("=" * 80)
print("\nInitial Proposal:")
print(f"  {result.initial_proposal.asp_rule}")
print(f"  Confidence: {result.initial_proposal.confidence:.2f}")
print(f"  Reasoning: {result.initial_proposal.reasoning}")

print("\nFinal Rule:")
print(f"  {result.final_rule.asp_rule}")
print(f"  Confidence: {result.final_rule.confidence:.2f}")
print(f"  Reasoning: {result.final_rule.reasoning}")

print("\nDebate Metrics:")
print(f"  Total Rounds: {result.total_rounds}")
print(f"  Converged: {result.converged}")
print(f"  Improvement: {result.improvement_score:+.2f}")
print(f"  Convergence Reason: {result.convergence_reason}")

# Get full transcript
print("\n" + "=" * 80)
print("FULL DIALECTICAL TRANSCRIPT")
print("=" * 80)
transcript = framework.get_debate_transcript(result)
print(transcript)

# Save to file
output_file = "debate_transcript_real.md"
with open(output_file, "w") as f:
    f.write("# Real Dialectical Debate Transcript\n\n")
    f.write("Generated using Claude 3.5 Sonnet via Anthropic API\n\n")
    f.write("## Summary\n\n")
    f.write(f"- **Initial Proposal**: {result.initial_proposal.asp_rule}\n")
    f.write(f"- **Final Rule**: {result.final_rule.asp_rule}\n")
    f.write(f"- **Total Rounds**: {result.total_rounds}\n")
    f.write(f"- **Converged**: {result.converged}\n")
    f.write(f"- **Improvement**: {result.improvement_score:+.2f}\n\n")
    f.write("## Full Transcript\n\n")
    f.write("```\n")
    f.write(transcript)
    f.write("\n```\n")

print(f"\n✅ Transcript saved to {output_file}")
