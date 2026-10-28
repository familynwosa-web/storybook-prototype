"""AI Storybook Generator

Generates a three-page children's story with Gemini and creates a matching
AI illustration for each page using Pollinations.ai.

Environment variables:
    GEMINI_API_KEY          Required for story generation.
    POLLINATIONS_API_KEY     Optional; use if your Pollinations account/API
                             plan requires a key. Never hard-code secrets.
"""

# Standard-library helpers: JSON parsing, environment variables, threading,
# queues for returning API results, elapsed-time tracking, URL encoding, and
# in-memory image inspection without creating temporary files.
import json
import os
import queue
import threading
import time
from io import BytesIO
from urllib.parse import quote

# External libraries: requests calls Pollinations.ai, Streamlit builds the UI,
# and google-genai provides the Gemini client used to write the story.
import requests
import streamlit as st
from google import genai
from google.genai import types

# ----------------------------- Configuration -----------------------------
st.set_page_config(
    page_title="AI Storybook Generator",
    page_icon="📖",
    layout="wide",
)

# Load the API key from the environment instead of hard-coding a secret.
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error(
        "GEMINI_API_KEY is missing. Set it as an environment variable "
        "or in your Streamlit secrets before running the app."
    )
    st.stop()

client = genai.Client(api_key=api_key)

# Pollinations image-generation endpoint. The Secret Key is loaded inside
# generate_pollinations_image() from the environment or Streamlit secrets.
# Keep API keys out of source code and never commit secrets to version control.
POLLINATIONS_IMAGE_ENDPOINT = "https://gen.pollinations.ai/image/"


# Try the preferred model first, then fall back to other currently listed
# Gemini Flash models if the service reports temporary overload/high demand.
# Try these Gemini models in order. Temporary overload errors trigger a switch
# to the next model; other errors are reported rather than blindly retried.
GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
]


def format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as MM:SS or H:MM:SS."""
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def run_with_live_timer(function, timer_placeholder, overall_start):
    """Run an API call in a worker while refreshing the visible timer.

    The background thread waits for the network request while the main thread
    periodically refreshes the elapsed-time display in Streamlit.
    """
    # The worker sends either the API result or its exception through this queue.
    result_queue = queue.Queue(maxsize=1)

    def worker():
        try:
            result_queue.put((True, function()))
        except BaseException as error:
            result_queue.put((False, error))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    # Refresh the on-screen timer until the background API call has finished.
    while True:
        timer_placeholder.markdown(
            f"**⏱️ Total run time:** {format_elapsed(time.monotonic() - overall_start)}"
        )
        try:
            succeeded, result = result_queue.get(timeout=0.4)
            break
        except queue.Empty:
            continue

    thread.join(timeout=0.1)
    timer_placeholder.markdown(
        f"**⏱️ Total run time:** {format_elapsed(time.monotonic() - overall_start)}"
    )
    if succeeded:
        return result
    raise result


def is_temporary_capacity_error(error: Exception) -> bool:
    """Recognize temporary errors where trying another model may help.

    Invalid API keys and malformed requests are not capacity errors, so they
    should be shown to the user instead of triggering model fallback.
    """
    message = str(error).lower()
    indicators = (
        "503", "429", "resource_exhausted", "unavailable", "high demand",
        "overloaded", "rate limit", "too many requests", "server busy",
        "temporarily unavailable", "internal server error",
    )
    return any(indicator in message for indicator in indicators)


def make_story_pages(
    theme: str,
    age: int,
    art_style: str,
    timer_placeholder,
    activity_placeholder,
    overall_start: float,
) -> tuple[list[dict], str]:
    """Generate structured page text, falling back to other models on overload."""
    # Ask for JSON so each page's title, story text, and image prompt can be
    # extracted separately and displayed in the correct place in the storybook.
    prompt = f"""
Create a warm, age-appropriate children's storybook in exactly 3 pages.
Theme: {theme}
Target reader age: {age}
Visual art style: {art_style}

Return ONLY valid JSON in this exact structure:
{{
  "pages": [
    {{
      "title": "Short page title",
      "text": "The story text for this page (2-4 short sentences)",
      "image_prompt": "Detailed visual prompt for illustrating this exact page"
    }}
  ]
}}

Requirements:
- Include exactly 3 objects in pages, in story order.
- Keep language, themes, and conflict appropriate for children aged {age}.
- Give the story a clear beginning, middle, and satisfying ending.
- Give the main characters the same appearance across all three image prompts.
- Each image_prompt must describe only the scene for its corresponding page.
- Describe a child-friendly storybook illustration in {art_style} style.
- No text, lettering, captions, logos, watermarks, frightening imagery, or violence in images.
- Do not include anything outside the JSON object.
"""

    # Attempt models in order. Only recognized temporary capacity errors cause
    # the loop to continue to the next model in the fallback list.
    response = None
    selected_model = None
    for model_index, model_name in enumerate(GEMINI_MODELS):
        activity_placeholder.info(
            f"Writing the story with Gemini model **{model_name}** "
            f"(attempt {model_index + 1} of {len(GEMINI_MODELS)})…"
        )
        try:
            response = run_with_live_timer(
                lambda model=model_name: client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                ),
                timer_placeholder,
                overall_start,
            )
            selected_model = model_name
            activity_placeholder.success(f"Story generated successfully using **{model_name}**.")
            break
        except Exception as error:
            has_next_model = model_index < len(GEMINI_MODELS) - 1
            if has_next_model and is_temporary_capacity_error(error):
                next_model = GEMINI_MODELS[model_index + 1]
                # Make the model switch visible so the user knows why waiting continues.
                st.warning(
                    f"**{model_name} is busy or experiencing high demand.** "
                    f"Switching to **{next_model}**. The app will continue automatically."
                )
                continue
            raise RuntimeError(f"Gemini model {model_name} failed: {error}") from error

    if response is None or selected_model is None:
        raise RuntimeError(
            "All configured Gemini models are currently busy or unavailable. "
            "Please wait a moment and try again."
        )

    # Parse and validate Gemini's JSON before trying to display the story pages.
    data = json.loads(response.text)
    pages = data.get("pages", [])
    if len(pages) != 3:
        raise ValueError("Gemini did not return exactly three story pages. Please try again.")

    for index, page in enumerate(pages, start=1):
        if not all(page.get(field) for field in ("title", "text", "image_prompt")):
            raise ValueError(f"Story page {index} is missing a title, story text, or image prompt.")
    return pages, selected_model

def describe_image_for_continuity(image_bytes: bytes, next_scene_prompt: str) -> str:
    """Use Gemini vision to inspect the last image and describe what must persist.

    The previous illustration is sent as image input to Gemini. Gemini returns
    a concise visual continuity guide for the next page: character appearance,
    clothing, proportions, palette, art treatment, and recurring environment.
    Pollinations' endpoint below is still text-to-image, so this guide is
    appended to the next prompt rather than passing raw image bytes to Pollinations.
    """
    # Detect common image formats from their file signatures so Gemini receives
    # the correct MIME type for the actual bytes returned by Pollinations.
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        mime_type = "image/png"
    elif image_bytes.startswith(b"\xff\xd8\xff"):
        mime_type = "image/jpeg"
    elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        mime_type = "image/webp"
    else:
        # Pollinations normally returns PNG or JPEG. Fail clearly if an
        # unexpected format arrives instead of sending invalid image input.
        raise ValueError("The previous illustration uses an unsupported image format.")

    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    continuity_prompt = f"""
Inspect the supplied children's storybook illustration as a visual reference
for the NEXT illustration.

Next page scene to illustrate:
{next_scene_prompt}

Return a concise continuity guide, not a new scene description. Carefully identify
only visible details you can actually observe:
- each character's species or type, eye color, fur/scale/feather colors, clothing,
  accessories, and distinctive markings (do NOT mention age, children, bodies, skin,
  or anatomy; refer to them only as storybook characters or creatures);
- the exact illustration medium, linework, texture, lighting, color palette, and
  level of detail;
- recurring location features, background, and environmental design.

Preserve those visible details in the next image. Do not invent details that
cannot be determined from the reference. The next page may change pose, action,
camera angle, and scene composition to match its story text, but character identity
and the overall visual design must remain recognizable. Keep the response brief
and suitable for appending to an image-generation prompt.
"""
    # Try each Gemini model in turn so one busy model doesn't block the page.
    last_error = None
    for model_name in GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[continuity_prompt, image_part],
            )
            description = (response.text or "").strip()
            if description:
                # The guide is embedded in a GET URL for Pollinations, so keep it
                # short and on one line to avoid oversized-URL failures.
                description = " ".join(description.split())
                return description[:700]
            last_error = RuntimeError("Gemini returned an empty continuity description.")
        except Exception as error:
            last_error = error
            if not is_temporary_capacity_error(error):
                break
    raise RuntimeError(f"Gemini could not describe the reference image: {last_error}")


def is_blocked_image(image_bytes: bytes) -> bool:
    """Return True if Pollinations sent back a 'Request blocked' placeholder.

    Pollinations answers safety-filter rejections with a normal image (HTTP 200),
    so the response looks like a success. Ask Gemini vision whether the picture
    is an error/blocked screen instead of a storybook illustration. If the check
    itself fails, assume the image is fine rather than blocking the story.
    """
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        mime_type = "image/png"
    elif image_bytes.startswith(b"\xff\xd8\xff"):
        mime_type = "image/jpeg"
    elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        mime_type = "image/webp"
    else:
        return False

    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    question = (
        "Is this image an error or moderation notice (for example a dialog saying "
        "'Request blocked' or 'Safety filter') rather than a real illustration? "
        "Answer with exactly one word: YES or NO."
    )
    for model_name in GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[question, image_part],
            )
            return (response.text or "").strip().upper().startswith("YES")
        except Exception as error:
            if not is_temporary_capacity_error(error):
                return False
    return False


def generate_pollinations_image(
    prompt: str,
    seed: int,
    continuity_guide: str = "",
) -> bytes:
    """Generate one illustration using the authenticated Pollinations API.

    The API key is read at call time so updated environment variables or
    Streamlit secrets are picked up without relying on a stale module-level
    value. Never log or display the secret itself.
    """
    api_key = os.getenv("POLLINATIONS_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("POLLINATIONS_API_KEY", "")
        except Exception:
            # Streamlit secrets may not be configured in local environments.
            api_key = ""

    api_key = api_key.strip() if api_key else ""
    if not api_key:
        raise RuntimeError(
            "Pollinations API key is missing. Set POLLINATIONS_API_KEY "
            "in your environment or Streamlit secrets. See "
            "https://enter.pollinations.ai/keys."
        )

    # Add consistent child-friendly style instructions to the page-specific
    # scene prompt supplied by Gemini.
    # For pages after page 1, continuity_guide was produced by visually
    # analyzing the immediately preceding illustration with Gemini vision.
    # Keep the page-specific action from the story, while strongly prioritizing
    # the reference image's character designs, palette, and art direction.
    continuity_section = (
        f"VISUAL CONTINUITY FROM THE PREVIOUS PAGE: {continuity_guide}. "
        "Treat these details as a strict design reference. Keep the same "
        "characters recognizable; do not redesign their faces, colors, clothing, "
        "body proportions, accessories, or the established illustration style. "
        if continuity_guide else
        "This is the first page; establish a clear, memorable character design "
        "and a consistent storybook art direction. "
    )
    full_prompt = (
        "Family-friendly illustrated picture book scene, visually appealing, "
        "gentle expressions, cheerful and safe. "
        f"{continuity_section}"
        f"SCENE FOR THIS PAGE: {prompt}. "
        "No text, no letters, no logos, no watermark."
    )
    url = POLLINATIONS_IMAGE_ENDPOINT + quote(full_prompt, safe="")

    # Request the selected image model, landscape dimensions, and page seed.
    params = {
        "model": "black-forest-labs/flux.1-schnell",
        "width": 1024,
        "height": 768,
        "seed": seed,
        "nologo": "true",
        "safe": "true",
    }

    # Pollinations authenticates this server-side request using a Secret Key.
    headers = {"Authorization": f"Bearer {api_key}"}

    # Image generation can take time, so allow a longer read timeout.
    result = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=(15, 180),
    )

    # Provide actionable messages for common authentication/access failures.
    # Do not include the request headers or API key in error messages.
    if result.status_code == 401:
        raise RuntimeError(
            "Pollinations returned 401 Unauthorized. Verify that "
            "POLLINATIONS_API_KEY is a valid Secret Key and that Streamlit "
            "has loaded it. Create or verify your key at "
            "https://enter.pollinations.ai/keys."
        )
    if result.status_code == 403:
        raise RuntimeError(
            "Pollinations denied access (403 Forbidden). Check the key's "
            "permissions and the API response for the specific reason."
        )
    if result.status_code == 402:
        raise RuntimeError(
            "Pollinations requires additional Pollen balance or a higher "
            "key spending budget. Check your Pollinations account."
        )

    # Raise for any other unsuccessful HTTP response.
    result.raise_for_status()

    # Reject JSON/text error responses that are not actual image data.
    content_type = result.headers.get("Content-Type", "")
    if not content_type.startswith("image/"):
        message = result.text[:300] if result.text else "No response body."
        raise RuntimeError(
            "Pollinations did not return an image "
            f"({content_type or 'unknown content type'}): {message}"
        )

    return result.content


# ------------------------------- Interface -------------------------------
# The sidebar collects preferences; the main page displays the story and images.
st.title("📖 AI Storybook Generator")
st.subheader("Executive MBA Prototype")
st.markdown("Generate a personalized three-page children's story with a matching AI illustration on every page.")

st.sidebar.header("Story Settings")
story_theme = st.sidebar.text_input("Story Theme", "A young dragon learns teamwork")
target_age = st.sidebar.slider("Target Age", min_value=3, max_value=12, value=3)
art_style = st.sidebar.selectbox(
    "Art Style",
    ["Watercolor", "Comic", "Pixel Art", "Claymation", "Fantasy"],
)
generate_button = st.sidebar.button("Generate Storybook", type="primary")

# Streamlit reruns the script after interactions. This block runs the workflow
# only on the run triggered by clicking the Generate Storybook button.
if generate_button:
    if not story_theme.strip():
        st.warning("Please enter a story theme first.")
        st.stop()

    # Use a monotonic clock to measure duration reliably, even if system time changes.
    overall_start = time.monotonic()
    st.success("Storybook generation started!")
    status_col, timer_col = st.columns([3, 1])
    with status_col:
        activity_placeholder = st.empty()
        activity_placeholder.info("Preparing your storybook…")
    with timer_col:
        timer_placeholder = st.empty()
        timer_placeholder.markdown("**⏱️ Total run time:** 00:00")

    with st.expander("Your Story Settings", expanded=False):
        st.write(f"**Theme:** {story_theme}")
        st.write(f"**Target age:** {target_age}")
        st.write(f"**Art style:** {art_style}")

    try:
        activity_placeholder.info("Generating the story. If a Gemini model is busy, the app will switch to the next model automatically.")
        # Generate structured story pages first. Each page includes its own image prompt.
        pages, selected_model = make_story_pages(
            story_theme,
            target_age,
            art_style,
            timer_placeholder,
            activity_placeholder,
            overall_start,
        )

        st.header("Your Storybook")
        st.caption(f"Story model used: {selected_model}")
        # Generate images sequentially. Each finished image is shown to Gemini
        # vision before the next page is generated, so the next prompt can reuse
        # observed character and environment details instead of relying only on
        # the story's original text prompts.
        previous_image_bytes = None
        continuity_guide = ""

        # Display pages in story order, with text and a matching illustration side by side.
        for page_number, page in enumerate(pages, start=1):
            st.divider()
            st.subheader(f"Page {page_number}: {page['title']}")
            text_col, image_col = st.columns([1, 1.15], gap="large")
            with text_col:
                st.write(page["text"])
            with image_col:
                activity_placeholder.info(f"Generating the illustration for page {page_number} using Pollinations.ai…")
                try:
                    # Before pages 2 and 3, send the immediately preceding image
                    # back to Gemini as multimodal input. Its visual analysis is
                    # then included in the next Pollinations generation prompt.
                    if previous_image_bytes is not None:
                        activity_placeholder.info(
                            f"Analyzing page {page_number - 1}'s illustration with Gemini "
                            "to preserve character and setting consistency…"
                        )
                        # If the analysis fails, still generate the image
                        # (without the guide) instead of losing the page.
                        try:
                            continuity_guide = run_with_live_timer(
                                lambda image=previous_image_bytes, next_prompt=page["image_prompt"]:
                                    describe_image_for_continuity(image, next_prompt),
                                timer_placeholder,
                                overall_start,
                            )
                        except Exception as guide_error:
                            continuity_guide = ""
                            st.info(
                                f"Couldn't analyze page {page_number - 1}'s image for continuity "
                                f"({guide_error}). Generating page {page_number} without it."
                            )

                    # The same timer covers image generation as well as story writing.
                    # Pollinations' safety filter can return a "Request blocked"
                    # picture as a normal image. Detect it and retry: first with
                    # the full guide, then a shorter guide and new seed, and
                    # finally with no guide at all.
                    base_seed = 20261027 + page_number
                    attempts = [
                        (continuity_guide, base_seed),
                        (continuity_guide[:300], base_seed + 100),
                        ("", base_seed + 200),
                    ]
                    image_bytes = None
                    for attempt_number, (guide, seed) in enumerate(attempts, start=1):
                        if attempt_number > 1:
                            activity_placeholder.info(
                                f"Page {page_number}'s image was blocked by the safety filter. "
                                f"Retrying with a simpler prompt (attempt {attempt_number} of {len(attempts)})…"
                            )
                        candidate = run_with_live_timer(
                            lambda prompt=page["image_prompt"], seed=seed,
                                   guide=guide: generate_pollinations_image(prompt, seed, guide),
                            timer_placeholder,
                            overall_start,
                        )
                        blocked = run_with_live_timer(
                            lambda data=candidate: is_blocked_image(data),
                            timer_placeholder,
                            overall_start,
                        )
                        if not blocked:
                            image_bytes = candidate
                            break
                    if image_bytes is None:
                        raise RuntimeError(
                            "Pollinations' safety filter blocked every attempt for this page. "
                            "Try a different theme or art style."
                        )
                    previous_image_bytes = image_bytes
                    st.image(
                        image_bytes,
                        caption=f"Page {page_number} illustration",
                        use_container_width=True,
                    )
                # Keep the story available even if one page's illustration fails.
                except requests.RequestException as image_error:
                    st.warning(
                        f"The story text for page {page_number} is ready, but Pollinations.ai "
                        f"could not generate its image. Please retry this story. Details: {image_error}"
                    )
                except Exception as image_error:
                    st.warning(f"Could not create the page {page_number} illustration: {image_error}")

        total_elapsed = format_elapsed(time.monotonic() - overall_start)
        timer_placeholder.markdown(f"**⏱️ Total run time:** {total_elapsed}")
        activity_placeholder.success(f"Storybook complete in {total_elapsed}.")
        st.caption(
            "For pages 2 and 3, Gemini visually analyzes the preceding illustration "
            "and adds a continuity guide to the next Pollinations prompt. Pollinations "
            "is still generating from text, not directly editing the prior image, so "
            "this improves consistency but cannot guarantee pixel-identical characters."
        )

    # Show a helpful message and the elapsed time if generation or parsing fails.
    except json.JSONDecodeError:
        elapsed = format_elapsed(time.monotonic() - overall_start)
        timer_placeholder.markdown(f"**⏱️ Total run time:** {elapsed}")
        activity_placeholder.error(f"Gemini returned invalid JSON after {elapsed}. Please try again.")
    except Exception as error:
        elapsed = format_elapsed(time.monotonic() - overall_start)
        timer_placeholder.markdown(f"**⏱️ Total run time:** {elapsed}")
        activity_placeholder.error(f"Couldn't generate the storybook after {elapsed}: {error}")