import streamlit as st
import requests
import json # For parsing LLM responses

# --- Configuration ---
# Secrets are expected to be set in Streamlit Cloud app settings
OPENROUTER_API_KEY_SECRET = st.secrets.get("OPENROUTER_API_KEY", "")
LLM_MODEL_SECRET = st.secrets.get("LLM_MODEL", "mistralai/mistral-7b-instruct:free") # A good default
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# --- Prompts ---
SYSTEM_PROMPT = """
You are "Holistic Health Insight Assistant," an AI designed to help patients gather comprehensive information about their long-term health issues for their doctor.
Your primary goal is to conduct a holistic assessment through a guided conversation, covering physiological and psychological aspects. This information will help the patient's doctor understand potential root causes and consider appropriate specialist departments or types of care.

Your interaction style:
1.  **One Question Per Turn:** IMPORTANT: In each of your responses, ask only ONE clear and concise primary question. Wait for the user's answer before asking the next question. Do not list multiple questions at once.
2.  **Empathetic & Patient:** Maintain an empathetic, patient, understanding, and non-judgmental tone. Use phrases like "I understand," "Thank you for sharing," "That sounds challenging," "Take your time."
3.  **No Medical Advice:** You must NOT provide medical advice, diagnoses, interpretations of symptoms, or treatment recommendations. If asked, politely state: "As an AI assistant, I'm here to help gather information for your doctor. I can't provide medical advice or diagnoses, but your doctor will be able to help with that."
4.  **Emergency Handling:** If the patient describes something that sounds like an acute medical emergency (e.g., sudden severe chest pain, difficulty breathing, active suicidal thoughts with a plan), gently and clearly advise them: "What you're describing sounds serious and may require immediate medical attention. Please contact your doctor, emergency services, or go to the nearest emergency department. As an AI, I cannot provide emergency help." Then, you can offer to pause the assessment.

Initial Information Gathering:
1.  Start by warmly introducing yourself and briefly explaining your purpose. Emphasize you are an AI assistant for information gathering.
2.  **Crucial First Steps:** Before diving into symptoms, you MUST ask the following, one at a time, using the JSON format specified below for open-ended text input:
    a. First, ask for age: {"question_text": "To begin, could you please tell me your current age?", "input_type": "text"} (Wait for response)
    b. After age, ask for gender: {"question_text": "Thank you. And what is your gender? (e.g., Male, Female, Non-binary, or how you identify)", "input_type": "text"} (Wait for response. You can also offer options for gender using the 'options' JSON format if you prefer).
3.  After getting age and gender, then ask: {"question_text": "Thank you for that information. Now, could you please tell me about the main health concerns or symptoms you'd like to discuss today?", "input_type": "text"}

Main Assessment Questions - How to ask and format:
1.  Systematically guide the conversation to explore different areas relevant to long-term health.
2.  **Timeline Memory:** When the user mentions symptoms or health events, actively try to understand the **timeline**. If a timeline is unclear, ask clarifying questions like, 'When did this particular symptom start?'
3.  **JSON for Options:**
    *   When asking a question where predefined options are suitable, **you MUST provide them in a specific JSON format.**
    *   **Each option MUST include a concise example.**
    *   The JSON structure MUST be exactly:
      ```json
      {
        "question_text": "Your single, clear question here.",
        "options": [
          {"value": "Option A Text", "example": "e.g., a brief clarifying example for Option A"},
          {"value": "Option B Text", "example": "e.g., example for Option B"},
          {"value": "Other (please specify)", "example": "Select this if your answer isn't listed."}
        ],
        "allow_multiple_selections": true,
        "input_type": "options"
      }
      ```
    *   Example for pain type (this is how *you* would format *your* response):
      ```json
      {
        "question_text": "How would you describe the pain you're feeling most often? You can select more than one if they apply.",
        "options": [
          {"value": "Aching", "example": "like a dull, constant muscle soreness"},
          {"value": "Throbbing", "example": "like a pulsing or beating sensation"},
          {"value": "Stabbing", "example": "like a sharp, sudden, piercing feeling"},
          {"value": "Burning", "example": "like a hot or searing sensation"},
          {"value": "Stiff", "example": "like your joints/muscles are tight, often worse in the morning"},
          {"value": "Other (please specify)", "example": "If none of these quite fit."}
        ],
        "allow_multiple_selections": true,
        "input_type": "options"
      }
      ```
4.  **Open-Ended Text Questions:**
    *   If a question is genuinely open-ended, format your response as:
      ```json
      {
        "question_text": "Your open-ended question here.",
        "input_type": "text"
      }
      ```
5.  **Clarity and Transitions:** Ask clarifying follow-up questions if needed (one at a time). Use transitions.

Concluding the Assessment:
1.  When ready to conclude, ask (using text JSON format):
    ```json
    {
      "question_text": "Is there anything else important about your health that you feel we haven't covered?",
      "input_type": "text"
    }
    ```
2.  After their response, if they have nothing more, conclude with (using a special type):
    ```json
    {
      "question_text": "Thank you for sharing all this information. When you're ready, click the 'Generate Doctor's Summary' button below. I'll then prepare a summary of our conversation for you to review and share with your doctor.",
      "input_type": "text_display_only"
    }
    ```
Start now: introduce yourself, then ask for age.
"""

SUMMARIZATION_PROMPT_TEMPLATE = """
Based on the following conversation with a patient (including their age and gender if provided at the start):
--- START OF CONVERSATION ---
{conversation_history_text}
--- END OF CONVERSATION ---

Please generate a structured summary for their doctor. The summary MUST:
1.  Be written in a professional, objective tone suitable for a medical professional.
2.  Start with "Patient Demographics:" including Age and Gender if available from the conversation.
3.  Then, "Patient's Primary Stated Health Concerns:" followed by a concise list or paragraph.
4.  Organize the rest of the information by key health domains discussed (e.g., Detailed Symptom Review, Digestive Health, Energy & Sleep, Mood & Stress, Pain Profile, Diet & Nutrition, Lifestyle Factors, Relevant Medical History, etc.). Under each domain, list the key symptoms, their characteristics (onset, duration, severity, frequency, triggers, alleviating factors), and any relevant details provided by the patient using bullet points for clarity.
5.  Include a section for "Potential Areas for Further Clinical Exploration:" This section should highlight any significant patterns, co-occurring symptoms, or potential system imbalances that emerge from the patient's narrative. Frame these as observations or hypotheses for the doctor to consider, NOT as diagnoses.
6.  Based on the summarized information, suggest 2-5 "Potential Referral or Consultation Pathways:" These should be types of medical departments, specialists, or allied health professionals. Briefly justify each suggestion by linking it to specific aspects of the patient's reported information.
7.  If mentioned, include a section for "Current Medications:" and "Significant Past Medical History:". If not mentioned, state "Not explicitly detailed by patient in this conversation."
8.  Conclude with a prominent disclaimer: "Disclaimer: This summary is based on patient self-reporting gathered through an AI-assisted conversational interface. It is intended for informational purposes to support clinical assessment and does NOT constitute a medical diagnosis. All clinical decisions, diagnostic interpretations, and treatment plans remain the sole responsibility of the attending physician based on their comprehensive clinical evaluation."

Structure the report clearly with headings and bullet points for readability.
Focus on factual reporting of what the patient stated.
"""

# --- LLM Interaction Function ---
def get_llm_response(conversation_history):
    if not OPENROUTER_API_KEY_SECRET:
        st.error("OpenRouter API Key is not configured in app secrets.")
        return None

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY_SECRET}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://your-app-name.streamlit.app", # Optional: Replace with your app URL if known
        # "X-Title": "Holistic Health Screener", # Optional
    }
    payload = {
        "model": LLM_MODEL_SECRET,
        "messages": conversation_history
    }
    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, data=json.dumps(payload), timeout=180)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error: {e}")
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        st.error(f"Error parsing LLM response: {e} - Response: {response.text if 'response' in locals() else 'No response object'}")
    return None

# --- Streamlit App UI and Logic ---
st.set_page_config(layout="wide", page_title="Holistic Health Screener")
st.title("🌿 Holistic Health Insight Assistant 🌿")

st.sidebar.header("About this Tool")
st.sidebar.info(
    """
    This tool uses an AI to help you gather your health information.
    - Answer the AI's questions. Some may offer clickable options.
    - When the AI indicates the assessment is concluding, or if you type "END ASSESSMENT", the process will move towards summarization.
    - A summary will be generated for you to review and share with your doctor.
    **This tool does NOT provide medical advice or diagnosis.**
    """
)
st.sidebar.markdown(f"**Using Model:** `{LLM_MODEL_SECRET}` (configured in secrets)")
st.sidebar.markdown("---")
if st.sidebar.button("Clear Chat & Restart Assessment"):
    st.session_state.clear() # Clears all session state
    st.rerun()

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_llm_question_data" not in st.session_state: # To store parsed JSON from LLM
    st.session_state.current_llm_question_data = None
if "assessment_phase_complete" not in st.session_state: # True when LLM sends "text_display_only"
    st.session_state.assessment_phase_complete = False
if "summary_generated" not in st.session_state:
    st.session_state.summary_generated = False
if "summary_text" not in st.session_state:
    st.session_state.summary_text = ""


# --- Helper function to process LLM response and update UI state ---
def process_llm_response(response_text):
    st.session_state.current_llm_question_data = None # Reset
    try:
        data = json.loads(response_text)
        if isinstance(data, dict) and "input_type" in data:
            st.session_state.current_llm_question_data = data
            if data["input_type"] == "text_display_only":
                st.session_state.assessment_phase_complete = True
        else: # Not the expected JSON structure, treat as plain text
            st.session_state.current_llm_question_data = {"question_text": response_text, "input_type": "text"}
    except json.JSONDecodeError: # LLM didn't return valid JSON
        st.session_state.current_llm_question_data = {"question_text": response_text, "input_type": "text"}
    
    st.session_state.messages.append({"role": "assistant", "content": st.session_state.current_llm_question_data["question_text"]})


# --- Initial greeting from LLM ---
if not st.session_state.messages and OPENROUTER_API_KEY_SECRET:
    with st.spinner("Holistic Health Insight Assistant is starting up..."):
        initial_response_text = get_llm_response(
            conversation_history=[{"role": "system", "content": SYSTEM_PROMPT}]
        )
    if initial_response_text:
        process_llm_response(initial_response_text)
    else: # Fallback if API fails on first try
        st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm ready to start. To begin, could you please tell me your current age?"})
        st.session_state.current_llm_question_data = {"question_text": "Hello! I'm ready to start. To begin, could you please tell me your current age?", "input_type": "text"}
    st.rerun()


# --- Display chat messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --- Handle user input based on current LLM question type ---
if OPENROUTER_API_KEY_SECRET and not st.session_state.assessment_phase_complete and not st.session_state.summary_generated:
    llm_q_data = st.session_state.current_llm_question_data

    if llm_q_data and llm_q_data["input_type"] == "options":
        # Display options form
        options_data = llm_q_data.get("options", [])
        allow_multiple = llm_q_data.get("allow_multiple_selections", False)
        
        with st.form(key="options_form"):
            st.markdown(llm_q_data["question_text"]) # Display the actual question text from parsed JSON
            
            formatted_options = [f"{opt['value']} (e.g., {opt['example']})" if 'example' in opt and opt['example'] else opt['value'] for opt in options_data]
            
            if allow_multiple:
                selections = st.multiselect("Select all that apply:", formatted_options, key="multiselect_options")
            else:
                selection = st.radio("Choose one:", formatted_options, key="radio_options")

            other_option_present = any("other (please specify)" in opt["value"].lower() for opt in options_data)
            other_text = ""
            if other_option_present:
                other_text = st.text_input("If 'Other', please specify:", key="other_text_input")

            submit_button = st.form_submit_button(label="Submit Answer")

            if submit_button:
                user_response_parts = []
                raw_selections = []

                if allow_multiple:
                    raw_selections = selections
                elif selection: # For radio
                    raw_selections = [selection]
                
                for raw_sel in raw_selections:
                    # Extract the original value part before "(e.g., ...)"
                    original_value = raw_sel.split(" (e.g.,")[0]
                    user_response_parts.append(original_value)
                    if "other (please specify)" in original_value.lower() and other_text:
                        user_response_parts[-1] = f"Other: {other_text}" # Replace "Other" with specified text
                
                if not user_response_parts and other_text: # Only "Other" was effectively chosen by typing
                     user_response_parts.append(f"Other: {other_text}")

                user_response_full = "; ".join(user_response_parts) if user_response_parts else "No specific option selected"
                if not user_response_parts and not other_text and other_option_present and any("other" in sel.lower() for sel in raw_selections):
                    user_response_full = "Selected 'Other' but did not specify."


                st.session_state.messages.append({"role": "user", "content": user_response_full})
                st.session_state.current_llm_question_data = None # Clear current question, expect new one

                with st.spinner("Thinking..."):
                    llm_response_text = get_llm_response(
                        conversation_history=[{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages
                    )
                if llm_response_text:
                    process_llm_response(llm_response_text)
                st.rerun()

    elif llm_q_data and llm_q_data["input_type"] == "text":
        # Use st.chat_input for text
        if prompt := st.chat_input("Your answer:", key="text_chat_input"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.current_llm_question_data = None # Clear current question

            if "END ASSESSMENT" in prompt.upper() and len(st.session_state.messages) > 3: # Allow manual end
                st.session_state.assessment_phase_complete = True
                # Add a final message from assistant to guide to summary button
                st.session_state.messages.append({"role": "assistant", "content": "Understood. When you're ready, please click the 'Generate Doctor's Summary' button."})
                st.rerun()

            else:
                with st.spinner("Thinking..."):
                    llm_response_text = get_llm_response(
                        conversation_history=[{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages
                    )
                if llm_response_text:
                    process_llm_response(llm_response_text)
                st.rerun()
    
    # If current_llm_question_data is None but not assessment_phase_complete, it means we are waiting for LLM (e.g. after initial load error)
    # Or if it's text_display_only, the input section is skipped.


# --- Generate Summary Button ---
if st.session_state.assessment_phase_complete and not st.session_state.summary_generated and OPENROUTER_API_KEY_SECRET:
    if st.button("Generate Doctor's Summary", type="primary"):
        with st.spinner("Generating summary for your doctor... This may take a moment."):
            conversation_for_summary = "\n".join(
                [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages]
            )
            summarization_full_prompt_for_llm = SUMMARIZATION_PROMPT_TEMPLATE.format(
                conversation_history_text=conversation_for_summary
            )
            summary_payload_messages = [
                {"role": "system", "content": "You are an AI assistant specialized in summarizing patient conversations into structured medical reports for doctors. Follow the user's instructions precisely."},
                {"role": "user", "content": summarization_full_prompt_for_llm}
            ]
            summary_text_response = get_llm_response(summary_payload_messages)

            if summary_text_response:
                st.session_state.summary_text = summary_text_response
                st.session_state.summary_generated = True
                st.success("Summary generated successfully!")
            else:
                st.error("Could not generate the summary. Please check API key and try again if needed.")
            st.rerun()

# --- Display Generated Summary ---
if st.session_state.summary_generated:
    st.markdown("---")
    st.subheader("📋 Summary for Your Doctor")
    st.markdown(
        "Please review this summary carefully. You can copy it or download it to share with your doctor. "
        "Remember, this is based on your self-reported information and is not a diagnosis."
    )
    st.text_area("Doctor's Summary:", value=st.session_state.summary_text, height=600, key="summary_display_area_final")
    st.download_button(
        label="Download Summary as Text File",
        data=st.session_state.summary_text.encode("utf-8"),
        file_name="holistic_health_summary.txt",
        mime="text/plain"
    )

elif not OPENROUTER_API_KEY_SECRET:
    st.error("OpenRouter API Key is not configured in your app's secrets. Please ask the app administrator to set it up.")
