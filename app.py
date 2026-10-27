"""
AI Storybook Generator
----------------------
A Streamlit prototype that collects story preferences from the user,
sends those preferences to Google's Gemini model, and displays the
generated children's story.
"""

# Import os so we can read environment variables, such as the API key.
import os

# Import the Google Gen AI SDK used to send prompts to Gemini.
from google import genai

# Read the Gemini API key from the environment.
# Keeping the key outside the source code helps avoid hard-coding secrets.
api_key = os.getenv("GEMINI_API_KEY")

# Stop the app with a clear message if the API key has not been configured.
if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing. "
        "Set the environment variable before running the app."
    )

# Create a Gemini client that will use the API key for requests.
client = genai.Client(api_key=api_key)

# Import Streamlit, which builds the app's web interface.
import streamlit as st

# Configure the browser tab and overall page layout.
# set_page_config should be called before most other Streamlit UI commands.
st.set_page_config(
    page_title="AI Storybook Generator",  # Text shown in the browser tab
    page_icon="📖",                       # Icon shown in the browser tab
    layout="wide"                         # Use a wide page layout
)

# Display the app's main heading and prototype subtitle.
st.title("📖 AI Storybook Generator")
st.subheader("Executive MBA Prototype")

# Add a short description beneath the heading.
st.markdown(
    """
    Generate personalized storybooks using AI.
    """
)

# -------------------------------------------------------------------
# Sidebar: collect the user's story preferences.
# -------------------------------------------------------------------

# Add a heading to the sidebar.
st.sidebar.header("Story Settings")

# Let the user enter the topic or theme for the story.
# The second argument is the default text shown in the input field.
story_theme = st.sidebar.text_input(
    "Story Theme",
    "A young dragon learns teamwork"
)

# Let the user choose the intended reader's age.
# The slider allows ages from 3 through 12 and starts at age 8.
target_age = st.sidebar.slider(
    "Target Age",
    min_value=3,
    max_value=12,
    value=8
)

# Let the user select the preferred illustration style from a list.
# The selected style is included in the prompt sent to Gemini.
art_style = st.sidebar.selectbox(
    "Art Style",
    [
        "Watercolor",
        "Comic",
        "Pixel Art",
        "Claymation",
        "Fantasy"
    ]
)

# Create a button that the user clicks to request a story.
# The value is True during the Streamlit run triggered by a button click.
generate_button = st.sidebar.button(
    "Generate Storybook"
)

# Only start the generation workflow when the button has been clicked.
if generate_button:

    # Show a confirmation message that the generation process has started.
    st.success("Storybook generation started!")

    # Display the choices the user made, so they can review the request.
    st.write("### User Inputs")
    st.write(f"Theme: {story_theme}")
    st.write(f"Target Age: {target_age}")
    st.write(f"Art Style: {art_style}")

    # Add a heading where the generated story will appear.
    st.write("### Story Output")

    # Catch errors from the API call or response handling so the app can
    # show an error message instead of stopping without a helpful explanation.
    try:

        # Send a prompt to Gemini asking it to create a three-page story.
        # The f-string inserts the user's selected theme, age, and art style.
        response = client.models.generate_content(
            model="gemini-3.8-flash",  # Gemini model used to generate the story
            contents=f"""
            Write a 3-page children's story.

            Theme: {story_theme}

            Audience Age:
            {target_age}

            Illustration Style:
            {art_style}

            Make the story educational and engaging.
            """
        )

        # Display the text returned by Gemini in the Streamlit app.
        st.write(response.text)

        # Show a placeholder for illustrations, which are not implemented yet.
        st.write("### Illustration Output")
        st.info("Image generation coming in Version 2")

    # If anything inside the try block fails, show the error details in the app.
    except Exception as e:
        st.error(f"Error generating story: {e}")
