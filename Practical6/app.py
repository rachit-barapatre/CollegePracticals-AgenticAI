import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from fpdf import FPDF


# =========================================================
# SETUP
# =========================================================

load_dotenv()

MODEL = "gemini-2.5-flash"


@st.cache_resource
def get_client():

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. "
            "Please check your .env file."
        )

    return genai.Client(api_key=api_key)


# =========================================================
# PLANNER AGENT
# =========================================================

def planner_agent(goal):

    client = get_client()

    prompt = f"""
You are a PLANNER AGENT.

Your only job is to break the goal below into
a short, numbered list of simple, executable steps.

Create 3 to 6 steps.

Do NOT solve the goal yourself.
Only output the numbered steps.

GOAL:
{goal}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3
        )
    )

    return response.text.strip()


# =========================================================
# PARSE STEPS
# =========================================================

def parse_steps(plan_text):

    steps = []

    for line in plan_text.split("\n"):

        line = line.strip()

        if not line:
            continue

        cleaned = line.lstrip(
            "0123456789.)- "
        ).strip()

        if cleaned:
            steps.append(cleaned)

    return steps


# =========================================================
# EXECUTOR AGENT
# =========================================================

def executor_agent(step, previous_results):

    client = get_client()

    context = "\n".join(
        f"Step {i + 1} result: {result}"
        for i, result in enumerate(previous_results)
    )

    if not context:
        context = "No previous steps yet."

    prompt = f"""
You are an EXECUTOR AGENT.

Carry out the single step below and give
a short, direct result.

Use the context from previous steps if relevant.

CONTEXT SO FAR:
{context}

STEP TO EXECUTE:
{step}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3
        )
    )

    return response.text.strip()


# =========================================================
# PDF TEXT CLEANING
# =========================================================

def safe_text(text):

    """
    Convert unsupported Unicode characters into
    characters supported by the default Helvetica font.
    """

    replacements = {

        "→": "->",
        "←": "<-",

        "–": "-",
        "—": "-",

        "’": "'",
        "‘": "'",

        "“": '"',
        "”": '"',

        "•": "-",

        "✓": "[OK]",
        "✔": "[OK]",

        "✗": "[X]",
        "❌": "[X]",
        "✅": "[OK]",

        "🎯": "",
        "🧠": "",
        "⚙": "",
        "🏁": "",
        "🚀": "",
        "📄": "",
        "➡": "->",

    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )

    # Helvetica supports Latin-1.
    # Replace anything else with ?
    return (
        text
        .encode("latin-1", "replace")
        .decode("latin-1")
    )


# =========================================================
# BUILD PDF
# =========================================================

def build_pdf(
    goal,
    steps,
    results
):

    pdf = FPDF(
        format="A4"
    )

    pdf.set_auto_page_break(
        auto=True,
        margin=18
    )

    pdf.add_page()


    # -----------------------------------------------------
    # TITLE
    # -----------------------------------------------------

    pdf.set_font(
        "Helvetica",
        "B",
        18
    )

    pdf.set_text_color(
        30,
        30,
        30
    )

    pdf.cell(
        0,
        12,
        "Multi-Agent Workflow Report",
        ln=True
    )


    # -----------------------------------------------------
    # DATE
    # -----------------------------------------------------

    pdf.set_font(
        "Helvetica",
        "",
        10
    )

    pdf.set_text_color(
        100,
        100,
        100
    )

    generated_time = datetime.now().strftime(
        "%B %d, %Y at %I:%M %p"
    )

    pdf.cell(
        0,
        6,
        f"Generated on {generated_time}",
        ln=True
    )

    pdf.ln(6)


    # -----------------------------------------------------
    # GOAL
    # -----------------------------------------------------

    pdf.set_font(
        "Helvetica",
        "B",
        13
    )

    pdf.set_text_color(
        20,
        20,
        20
    )

    pdf.cell(
        0,
        8,
        "Goal",
        ln=True
    )

    pdf.set_font(
        "Helvetica",
        "",
        11
    )

    pdf.multi_cell(
        0,
        6,
        safe_text(goal)
    )

    pdf.ln(6)


    # -----------------------------------------------------
    # PLAN
    # -----------------------------------------------------

    pdf.set_font(
        "Helvetica",
        "B",
        13
    )

    pdf.cell(
        0,
        8,
        "Planner Agent - Generated Plan",
        ln=True
    )

    pdf.ln(3)

    pdf.set_font(
        "Helvetica",
        "",
        10
    )

    for i, step in enumerate(
        steps,
        1
    ):

        pdf.multi_cell(
            0,
            6,
            safe_text(
                f"{i}. {step}"
            )
        )

        pdf.ln(1)

    pdf.ln(6)


    # -----------------------------------------------------
    # EXECUTOR RESULTS
    # -----------------------------------------------------

    pdf.set_font(
        "Helvetica",
        "B",
        13
    )

    pdf.cell(
        0,
        8,
        "Executor Agent - Results",
        ln=True
    )

    pdf.ln(3)


    for i, (
        step,
        result
    ) in enumerate(
        zip(steps, results),
        1
    ):

        # Step

        pdf.set_font(
            "Helvetica",
            "B",
            11
        )

        pdf.multi_cell(
            0,
            6,
            safe_text(
                f"Step {i}: {step}"
            )
        )

        pdf.ln(1)


        # Result

        pdf.set_font(
            "Helvetica",
            "",
            10
        )

        pdf.multi_cell(
            0,
            6,
            safe_text(result)
        )

        pdf.ln(5)


    # -----------------------------------------------------
    # FOOTER
    # -----------------------------------------------------

    pdf.set_font(
        "Helvetica",
        "I",
        9
    )

    pdf.set_text_color(
        100,
        100,
        100
    )

    pdf.multi_cell(
        0,
        5,
        "Generated using Planner Agent + Executor Agent."
    )


    # =====================================================
    # IMPORTANT FIX
    # =====================================================

    # fpdf2 returns a bytearray when output() is used
    # directly. Convert bytearray to bytes.

    output = pdf.output()

    if isinstance(output, bytearray):

        return bytes(output)

    elif isinstance(output, bytes):

        return output

    else:

        # Fallback for versions returning string
        return output.encode("latin-1")


# =========================================================
# STREAMLIT CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Planner-Executor Agent Workflow",
    page_icon="🤖",
    layout="centered"
)


# =========================================================
# CUSTOM UI
# =========================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1.5rem;
    }

    .agent-card {
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        background-color: #fafafa;
    }

    .result-box {
        border-left: 3px solid #1f6feb;
        padding: 0.7rem 0.9rem;
        background-color: #f5f8ff;
        border-radius: 4px;
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.title(
    "🤖 Planner → Executor Agent Workflow"
)

st.caption(
    "A two-agent system where one agent plans "
    "and another agent executes each step."
)


# =========================================================
# HOW IT WORKS
# =========================================================

with st.expander(
    "How this works",
    expanded=False
):

    st.markdown(
        """
### Multi-Agent Workflow

**1. User**

Enter a high-level goal.

**2. Planner Agent**

Breaks the goal into 3-6 executable steps.

**3. Executor Agent**

Executes each step one-by-one.

**4. Final Summary**

Displays all generated results.

**5. PDF Export**

Download the complete workflow report.
"""
    )


st.divider()


# =========================================================
# SESSION STATE
# =========================================================

if "steps" not in st.session_state:

    st.session_state.steps = []


if "results" not in st.session_state:

    st.session_state.results = []


if "goal" not in st.session_state:

    st.session_state.goal = ""


if "finished" not in st.session_state:

    st.session_state.finished = False


# =========================================================
# USER INPUT
# =========================================================

with st.form("goal_form"):

    goal_input = st.text_area(
        "🎯 Enter a goal",
        placeholder=(
            "Example: Plan a 3-day trip to Goa "
            "on a budget"
        ),
        height=100
    )

    submitted = st.form_submit_button(
        "🚀 Run Workflow",
        use_container_width=True
    )


# =========================================================
# RUN WORKFLOW
# =========================================================

if submitted:

    if not goal_input.strip():

        st.warning(
            "Please enter a goal before running "
            "the workflow."
        )

    else:

        st.session_state.goal = (
            goal_input.strip()
        )

        st.session_state.steps = []

        st.session_state.results = []

        st.session_state.finished = False


        # =================================================
        # PLANNER
        # =================================================

        with st.status(
            "🧠 Planner Agent is creating a plan...",
            expanded=True
        ) as status:

            try:

                plan_text = planner_agent(
                    st.session_state.goal
                )

                steps = parse_steps(
                    plan_text
                )

                if not steps:

                    st.error(
                        "Planner did not generate "
                        "any steps."
                    )

                    st.stop()


                st.session_state.steps = steps


                st.write(
                    "**Generated Plan:**"
                )


                for i, step in enumerate(
                    steps,
                    1
                ):

                    st.write(
                        f"{i}. {step}"
                    )


                status.update(
                    label="✅ Plan created",
                    state="complete"
                )


            except Exception as e:

                status.update(
                    label="❌ Planner failed",
                    state="error"
                )

                st.error(
                    f"Planner error: {e}"
                )

                st.stop()


        # =================================================
        # EXECUTOR
        # =================================================

        st.subheader(
            "⚙️ Executor Agent"
        )


        results = []


        for i, step in enumerate(
            steps,
            1
        ):

            st.markdown(
                f"""
                <div class="agent-card">
                    <b>Step {i}</b><br>
                    {step}
                </div>
                """,
                unsafe_allow_html=True
            )


            with st.spinner(
                f"Executing Step {i}..."
            ):

                try:

                    result = executor_agent(
                        step,
                        results
                    )

                except Exception as e:

                    result = (
                        f"Execution failed: {e}"
                    )


            st.markdown(
                '<div class="result-box">',
                unsafe_allow_html=True
            )

            st.write(
                result
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


            results.append(
                result
            )


        st.session_state.results = results

        st.session_state.finished = True


# =========================================================
# FINAL SUMMARY
# =========================================================

if (
    st.session_state.finished
    and st.session_state.steps
):

    st.divider()


    st.subheader(
        "🏁 Final Summary"
    )


    for i, (
        step,
        result
    ) in enumerate(
        zip(
            st.session_state.steps,
            st.session_state.results
        ),
        1
    ):

        st.markdown(
            f"### Step {i}: {step}"
        )

        st.write(
            result
        )


    # =====================================================
    # CREATE PDF
    # =====================================================

    pdf_bytes = build_pdf(
        st.session_state.goal,
        st.session_state.steps,
        st.session_state.results
    )


    st.divider()


    # =====================================================
    # DOWNLOAD BUTTON
    # =====================================================

    st.download_button(
        label="📄 Download Workflow Report as PDF",
        data=pdf_bytes,
        file_name=(
            "agent_workflow_report_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        ),
        mime="application/pdf",
        use_container_width=True
    )