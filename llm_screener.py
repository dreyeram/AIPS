import streamlit as st
import requests # To make API calls to OpenRouter.ai
import json

# --- Configuration ---
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY_HERE") # Replace with your key or use Streamlit secrets
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
# Choose a model available on OpenRouter, e.g., "mistralai/mistral-7b-instruct" or a more capable one
# Check OpenRouter docs for available models: https://openrouter.ai/docs#models
LLM_MODEL = st.secrets.get("LLM_MODEL", "mistralai/mistral-7b-instruct:free") # Example: Use a free model for testing

# --- Prompts ---
SYSTEM_PROMPT = """
You are "Holistic Health Insight Assistant," an AI designed to help patients gather comprehensive information about their long-term health issues for their doctor.
Your primary goal is to conduct a holistic assessment through a guided conversation, covering physiological and psychological aspects. This information will help the patient's doctor understand potential root causes and consider appropriate specialist departments or types of care.

Your process for this conversation:
1.  Start by warmly introducing yourself and briefly explaining the purpose of this conversation: to gather information for their doctor.
2.  Ask open-ended questions to understand the patient's main health concerns and symptoms.
3.  Systematically and gently guide the conversation to explore different areas relevant to long-term health. These include, but are not limited to:
    *   Detailed description of primary symptoms (nature, frequency, severity, duration, triggers, what makes it better/worse).
    *   General medical history (major illnesses, surgeries, current medications - briefly).
    *   Digestive health (e.g., bloating, pain, bowel habits, food reactions).
    *   Energy levels and fatigue patterns.
    *   Sleep patterns and quality.
    *   Mood, stress levels, anxiety, and overall mental well-being.
    *   Pain (if any - location, type, frequency, intensity).
    *   Dietary habits and nutrition (typical meals, hydration, known sensitivities).
    *   Physical activity levels and types.
    *   Lifestyle factors (e.g., smoking, alcohol, caffeine).
    *   For women: menstrual cycle details, pregnancies, menopause, if relevant.
    *   Known allergies.
    *   Any perceived environmental factors impacting health.
4.  For each area, ask clarifying follow-up questions as needed. Be curious and thorough but also respectful of the patient's pace.
5.  Maintain an empathetic, patient, understanding, and non-judgmental tone throughout the conversation. Use phrases like "I understand," "Thank you for sharing," "That sounds challenging."
6.  IMPORTANT: You must NOT provide medical advice, diagnoses, interpretations of symptoms, or treatment recommendations. If asked, politely state that your role is to gather information for their doctor, who will provide medical guidance.
7.  If the patient describes something that sounds like an acute medical emergency (e.g., chest pain, difficulty breathing, suicidal thoughts), gently advise them to contact their doctor or emergency services immediately, while clearly stating you are an AI and cannot provide emergency help.
8.  Keep your questions clear and reasonably concise. Avoid asking multiple complex questions in a single turn.
9.  Let the conversation flow naturally. After discussing an area, you can transition by saying something like, "Thank you for sharing that. Now, I'd like to ask a bit about [next area]..."
10. When you feel you have gathered a good amount of information across several key areas, or if the patient indicates they have shared all primary concerns, you can ask, "Is there anything else important about your health that you feel we haven't covered or that you'd like to share with your doctor?"
11. To conclude the information-gathering phase, you can say something like: "Thank you so much for sharing all this information. I will now prepare a summary for you to review and then share with your doctor. This will help them in understanding your health situation more comprehensively." (The actual summary generation will be a separate step after this conversation).

Remember, your sole purpose in this phase is to ask questions and gather information.
Start the conversation now by introducing yourself and asking about their main health concerns.
"""

SUMMARIZATION_PROMPT_TEMPLATE = """
Based on the following conversation with a patient:
--- START OF CONVERSATION ---
{conversation_history_text}
--- END OF CONVERSATION ---

Please generate a structured summary for their doctor. The summary MUST:
1.  Be written for a medical professional.
2.  Start with the patient's stated primary health concerns.
3.  Organize the information by key health domains discussed (e.g., Digestive Health, Sleep Patterns, Mood & Stress, Pain, Diet & Lifestyle, etc.). Under each domain, list the key symptoms, their characteristics (onset, duration, severity, frequency), and any relevant details provided by the patient.
4.  Highlight any significant patterns, co-occurring symptoms, or potential areas of system imbalance that emerge from the patient's narrative. Frame these as observations or areas for potential further investigation, not diagnoses.
5.  Based on the summarized information, suggest 2-5 types of medical departments, specialists, or allied health professionals (e.g., Gastroenterologist, Neurologist, Endocrinologist, Dietitian, Psychologist, Physiotherapist, Sleep Specialist) that the doctor might consider for consultation or referral. Briefly justify each suggestion by linking it to specific aspects of the patient's reported information.
6.  List any current medications or major past medical history mentioned by the patient.
7.  Conclude with a prominent disclaimer: "This summary is based on patient self-reporting gathered through an AI-assisted conversational interface. It is intended for informational purposes to support clinical assessment and does NOT constitute a medical diagnosis. All clinical decisions, diagnostic interpretations, and treatment plans remain the sole responsibility of the attending physician."

Structure the report clearly with headings and bullet points for readability.
Avoid conversational language in the summary; it should be a concise, professional report.
"""


# --- LLM Interaction Function ---
def get_llm_response(conversation_history, prompt_to_use_for_llm, is_json_mode=False):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    # For OpenRouter, the "system" prompt is often best placed as the first message in the `messages` list with role "system"
    # Subsequent messages are "user" and "assistant"
    
    payload_messages = []
    if not conversation_history: # First message in a new assessment
        payload_messages.append({"role": "system", "content": prompt_to_use_for_llm}) # The system prompt itself
        # We might add a first "assistant" message to kickstart, or let the system prompt do it.
        # For assessment, the system prompt guides the *first* assistant message.
    else:
        # The full conversation history should already include the system prompt as its first element
        payload_messages = conversation_history


    payload = {
        "model": LLM_MODEL,
        "messages": payload_messages
    }
    if is_json_mode: # Some models support a JSON mode for structured output
        payload["response_format"] = {"type": "json_object"}


    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, data=json.dumps(payload), timeout=180) # Increased timeout
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error: {e}")
        return None
    except (KeyError, IndexError) as e:
        st.error(f"Error parsing LLM response: {e} - Response: {response.text}")
        return None

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Holistic Health Screener")
st.title("🌿 Holistic Health Insight Assistant 🌿")
st.markdown("I'm an AI assistant here to help you gather information about your health for your doctor.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "assessment_complete" not in st.session_state:
    st.session_state.assessment_complete = False
if "summary_generated" not in st.session_state:
    st.session_state.summary_generated = False


# Display initial greeting from LLM if chat is empty
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("Loading initial greeting...")
    # This is the first call to the LLM, it will use the SYSTEM_PROMPT
    initial_greeting = get_llm_response(
        conversation_history=[{"role": "system", "content": SYSTEM_PROMPT}], # Pass system prompt for LLM to generate its first turn
        prompt_to_use_for_llm=SYSTEM_PROMPT # Not strictly needed here if first call is handled this way
    )
    if initial_greeting:
        st.session_state.messages.append({"role": "assistant", "content": initial_greeting})
    else:
        st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm ready to start our conversation about your health. To begin, could you tell me about your main health concerns?"}) # Fallback
    # Re-run to display the greeting
    st.rerun()


# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input for user
if not st.session_state.assessment_complete:
    if prompt := st.chat_input("What would you like to share? (Type 'END ASSESSMENT' when done)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if "END ASSESSMENT" in prompt.upper() and len(st.session_state.messages) > 3 : # Added minimum length
            st.session_state.assessment_complete = True
            st.success("Assessment information gathering complete. Click the button below to generate the summary for your doctor.")
            st.rerun()
        else:
            # Prepare the full conversation history for the LLM
            # The first message should ideally be the system prompt.
            current_conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response_text = get_llm_response(current_conversation_history, SYSTEM_PROMPT) # System prompt provides ongoing context
                    if response_text:
                        st.markdown(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                    else:
                        st.markdown("Sorry, I encountered an issue. Please try again.")
                        # Don't add a failed response to history or handle appropriately

# Button to generate summary
if st.session_state.assessment_complete and not st.session_state.summary_generated:
    if st.button("Generate Doctor's Summary"):
        with st.spinner("Generating summary for your doctor... This may take a moment."):
            # Prepare conversation text for the summarization prompt
            conversation_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages if msg['role'] != 'system'])
            
            summarization_full_prompt = SUMMARIZATION_PROMPT_TEMPLATE.format(conversation_history_text=conversation_text)
            
            # For summary, we send the summarization prompt as a user message, with a minimal system prompt if needed by model
            # or treat the whole thing as the "user" part of a user/assistant pair.
            # The get_llm_response needs to be flexible or we need a dedicated one.
            # For simplicity, sending it as a "user" prompt to the LLM.
            # Constructing messages for summarization:
            summary_payload_messages = [
                {"role": "system", "content": "You are a helpful AI assistant tasked with summarizing a patient-AI conversation into a structured report for a doctor."}, # A general system prompt for this task
                {"role": "user", "content": summarization_full_prompt}
            ]

            summary_text = get_llm_response(summary_payload_messages, summarization_full_prompt) # prompt_to_use_for_llm is not strictly used if history is structured like this

            if summary_text:
                st.session_state.summary_text = summary_text
                st.session_state.summary_generated = True
                st.success("Summary generated!")
                st.rerun() # Rerun to display the summary
            else:
                st.error("Could not generate the summary.")

# Display the summary
if st.session_state.summary_generated:
    st.markdown("---")
    st.subheader("Summary for Your Doctor")
    st.markdown(st.session_state.get("summary_text", "Error: Summary not available."))
    st.download_button(
        label="Download Summary as Text File",
        data=st.session_state.get("summary_text", "").encode("utf-8"),
        file_name="holistic_health_summary.txt",
        mime="text/plain"
    )
    st.info("Please review this summary carefully. You can then download it and share it with your doctor.")

st.sidebar.header("About this Tool")
st.sidebar.info(
    """
    This tool uses a Large Language Model (AI) to help you gather information about your health.
    -   The AI will ask you questions conversationally.
    -   Answer honestly and thoroughly.
    -   When you feel you've shared enough, type "END ASSESSMENT".
    -   A summary will then be generated for you to review and share with your doctor.
    **This tool does NOT provide medical advice or diagnosis.**
    """
)
st.sidebar.markdown("---")
st.sidebar.text_input("Enter OpenRouter API Key (Optional)", key="OPENROUTER_API_KEY_INPUT", type="password", help="Overrides default if set")
if st.session_state.OPENROUTER_API_KEY_INPUT:
    OPENROUTER_API_KEY = st.session_state.OPENROUTER_API_KEY_INPUT
st.sidebar.text_input("Enter LLM Model (Optional)", key="LLM_MODEL_INPUT", help="e.g., mistralai/mistral-7b-instruct:free")
if st.session_state.LLM_MODEL_INPUT:
    LLM_MODEL = st.session_state.LLM_MODEL_INPUT


st.sidebar.markdown(f"**Using Model:** `{LLM_MODEL}`")
if OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE" or not OPENROUTER_API_KEY:
    st.sidebar.error("OpenRouter API Key not set. Please add it to Streamlit secrets or enter above.")
