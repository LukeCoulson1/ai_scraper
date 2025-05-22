import pandas as pd
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from bs4 import BeautifulSoup
from llama_cpp import Llama
import time
from urllib.parse import urlparse, urljoin
import re
import os
import io
import json
import tempfile
import streamlit as st
import requests
import random

try:
    from langdetect import detect
except ImportError:
    detect = None

# --- Session state for templates, history, and results ---
if "css_templates" not in st.session_state:
    st.session_state.css_templates = {}
if "prompt_templates" not in st.session_state:
    st.session_state.prompt_templates = {}
if "scrape_history" not in st.session_state:
    st.session_state.scrape_history = {}
if "results" not in st.session_state:
    st.session_state.results = {}
if "column_map" not in st.session_state:
    st.session_state.column_map = {}
if "user" not in st.session_state:
    st.session_state.user = "guest"
if "schedule" not in st.session_state:
    st.session_state.schedule = None
if "dashboard" not in st.session_state:
    st.session_state.dashboard = []

PROMPT_TEMPLATES = {
    "Summarize": "Summarize the following content.",
    "Extract Table": "Extract all tables from the following content in markdown format.",
    "Translate": "Translate the following content to {lang}.",
    "Extract Keywords": "Extract the most important keywords from the following content.",
    "Smart Extract": "Extract all structured data (tables, lists, key-value pairs, important facts) from the following content in markdown or JSON format.",
    "Entities": "Extract all emails, phone numbers, dates, and named entities from the following content.",
    "Explain": "Explain why you extracted the following data and show the relevant context.",
    "Custom": None
}

def scrape_page(url, css_selector=None, screenshot_path=None, headless=True):
    edge_options = Options()
    if headless:
        edge_options.add_argument("--headless")
        edge_options.add_argument("--disable-gpu")
    edge_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")
    edge_options.add_argument("--window-size=1920,1080")
    edge_options.add_argument("--disable-blink-features=AutomationControlled")
    service = Service(r"F:\Python_Programs\Dependencies\edgedriver_win64\msedgedriver.exe")
    driver = webdriver.Edge(service=service, options=edge_options)

    time.sleep(random.uniform(2, 5))  # Initial pause before loading the page

    driver.get(url)
    time.sleep(random.uniform(3, 7))  # Pause after page load

    if screenshot_path:
        driver.save_screenshot(screenshot_path)

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    driver.quit()

    if css_selector:
        section = soup.select_one(css_selector)
        if not section:
            section = soup
    else:
        section = soup.find('div', class_='product-details')
        if not section:
            section = soup
    text = section.get_text(separator='\n', strip=True)
    return text, str(section)

def extract_html_tables(html, base_url=None):
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all('table')
    dfs = []
    for table in tables:
        # Remove all <img> tags from the table before parsing
        table_soup = BeautifulSoup(str(table), 'html.parser')
        for img in table_soup.find_all('img'):
            img.decompose()
        try:
            df = pd.read_html(str(table_soup), flavor='bs4')[0]
            dfs.append(df)
        except Exception:
            continue
    return dfs

def gemma_llm_task(context, instruction, model_path, n_ctx=4096, max_tokens=1024):
    max_context_chars = 4000
    if len(context) > max_context_chars:
        context = context[:max_context_chars]
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=8,
        n_gpu_layers=-1
    )
    prompt = f"{instruction}\n\nContent:\n{context}\n\nResponse:"
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        stop=["Response:"]
    )
    return output["choices"][0]["text"].strip()

def url_to_filename(url, ext="txt"):
    parsed = urlparse(url)
    domain = parsed.netloc.replace('.', '_')
    path = re.sub(r'[^a-zA-Z0-9]', '_', parsed.path.strip('/'))
    if path:
        filename = f"{domain}_{path}.{ext}"
    else:
        filename = f"{domain}.{ext}"
    return filename

def extract_markdown_tables(text, reference_columns=None):
    lines = text.splitlines()
    tables = []
    table_lines = []
    in_table = False
    for line in lines + [""]:
        if re.match(r'^\s*\|', line):
            in_table = True
            table_lines.append(line.strip())
        elif in_table and not line.strip():
            if table_lines:
                if len(table_lines) > 1 and re.match(r'^\s*\|[\s\-|]+\|$', table_lines[1]):
                    del table_lines[1]
                data = [row.strip('|').split('|') for row in table_lines]
                try:
                    if reference_columns is not None:
                        if len(data) > 1 and [c.strip() for c in data[0]] == reference_columns:
                            df = pd.DataFrame(data[1:], columns=reference_columns)
                        else:
                            df = pd.DataFrame(data, columns=reference_columns)
                    else:
                        columns = [c.strip() for c in data[0]]
                        df = pd.DataFrame(data[1:], columns=columns)
                    tables.append(df)
                except Exception:
                    pass
                table_lines = []
            in_table = False
    return tables

def chunk_text(text, max_chars=4000):
    for i in range(0, len(text), max_chars):
        yield text[i:i + max_chars]

def detect_language(text):
    if detect:
        try:
            return detect(text)
        except Exception:
            return "unknown"
    return "unknown"

def search_keywords(text, keywords):
    found = []
    for kw in keywords:
        if re.search(kw, text, re.IGNORECASE):
            found.append(kw)
    return found

def save_tables_download(tables, save_type, merge=True, separate_excel_sheets=False):
    if not tables:
        st.warning("No tables found to save.")
        return None, None
    if merge:
        final_df = pd.concat(tables, ignore_index=True).drop_duplicates()
        if st.session_state.column_map:
            final_df = final_df.rename(columns=st.session_state.column_map)
        if save_type == "Excel":
            output = io.BytesIO()
            final_df.to_excel(output, index=False)
            return output.getvalue(), "result.xlsx"
        elif save_type == "CSV":
            return final_df.to_csv(index=False).encode(), "result.csv"
        elif save_type == "JSON":
            return final_df.to_json(orient="records", force_ascii=False).encode(), "result.json"
        elif save_type == "Markdown":
            return final_df.to_markdown(index=False).encode(), "result.md"
    elif separate_excel_sheets and save_type == "Excel":
        output = io.BytesIO()
        with pd.ExcelWriter(output) as writer:
            for idx, df in enumerate(tables):
                sheet_name = f"Table_{idx+1}"
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        return output.getvalue(), "result_multi.xlsx"
    else:
        st.warning("Separate download for non-Excel not supported in this demo.")
        return None, None

def plot_table(df):
    st.subheader("Quick Data Visualization")
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) >= 1:
        chart_type = st.selectbox("Chart type", ["Bar", "Line", "Area"], key="chart_type")
        col = st.selectbox("Select numeric column to plot", numeric_cols, key="chart_col")
        if chart_type == "Bar":
            st.bar_chart(df[col])
        elif chart_type == "Line":
            st.line_chart(df[col])
        elif chart_type == "Area":
            st.area_chart(df[col])
    else:
        st.info("No numeric columns to plot.")

def compare_with_previous(url, new_content):
    prev = st.session_state.scrape_history.get(url)
    changed = False
    if prev:
        st.markdown("#### Content Change Detection")
        import difflib
        diff = difflib.unified_diff(prev.splitlines(), new_content.splitlines(), lineterm='')
        diff_text = "\n".join(diff)
        if diff_text:
            st.code(diff_text, language="diff")
            changed = True
        else:
            st.success("No changes detected since last scrape.")
    st.session_state.scrape_history[url] = new_content
    return changed

def save_template(name, value, template_type="css"):
    user = st.session_state.user
    key = f"{user}:{name}"
    if template_type == "css":
        st.session_state.css_templates[key] = value
    else:
        st.session_state.prompt_templates[key] = value

def load_template(template_type="css"):
    user = st.session_state.user
    templates = {k: v for k, v in (st.session_state.css_templates if template_type == "css" else st.session_state.prompt_templates).items() if k.startswith(f"{user}:")}
    if templates:
        name = st.selectbox(f"Load {template_type.upper()} template", [k.split(":",1)[1] for k in templates.keys()])
        return templates[f"{user}:{name}"]
    else:
        st.info(f"No saved {template_type.upper()} templates.")
        return ""

def suggest_css_selector(html, user_text):
    soup = BeautifulSoup(html, 'html.parser')
    el = soup.find(string=re.compile(re.escape(user_text), re.I))
    if el and el.parent:
        return el.parent.name + (f".{el.parent.get('class')[0]}" if el.parent.get('class') else "")
    return ""

def extract_entities_llm(content, model_path, max_tokens=512):
    instruction = PROMPT_TEMPLATES["Entities"]
    return gemma_llm_task(content, instruction, model_path, max_tokens=max_tokens)

def explain_extraction_llm(content, extracted, model_path, max_tokens=512):
    instruction = PROMPT_TEMPLATES["Explain"] + f"\n\nExtracted Data:\n{extracted}"
    return gemma_llm_task(content, instruction, model_path, max_tokens=max_tokens)

def find_next_page_url(html, base_url):
    soup = BeautifulSoup(html, 'html.parser')
    next_link = soup.find('a', string=re.compile(r'next', re.I))
    if next_link and next_link.get('href'):
        return urljoin(base_url, next_link['href'])
    return None

def export_to_google_sheets(df, sheet_name, gsheet_url):
    st.info(f"To import to Google Sheets, download as CSV and use [File > Import] in your sheet: {gsheet_url}")

def post_to_webhook(data, webhook_url):
    try:
        resp = requests.post(webhook_url, json=data, timeout=10)
        if resp.status_code == 200:
            st.success("Results sent to webhook successfully!")
        else:
            st.warning(f"Webhook returned status {resp.status_code}")
    except Exception as e:
        st.error(f"Webhook error: {e}")

# --- User Dashboard ---
def show_dashboard():
    st.sidebar.markdown("### User Dashboard")
    if st.session_state.dashboard:
        for entry in st.session_state.dashboard[-10:][::-1]:
            st.sidebar.markdown(f"- [{entry['url']}]({entry['url']}) at {entry['time']} ({entry['status']})")
    else:
        st.sidebar.info("No scraping history yet.")

st.set_page_config(page_title="AI Scraper Web UI", layout="wide")
st.title("AI Scraper Web UI")

# --- Team/Collab & Scheduling ---
st.sidebar.markdown("### Team & Scheduling")
st.session_state.user = st.sidebar.text_input("Your username (for team/collab)", value=st.session_state.user)
webhook_url = st.sidebar.text_input("Webhook URL (optional, POST results after scrape)")
gsheet_url = st.sidebar.text_input("Google Sheet URL (optional, for export tip)")
if st.sidebar.button("Schedule scrape for next app run"):
    st.session_state.schedule = True
    st.sidebar.success("Scrape scheduled! It will run on next app reload.")

show_dashboard()

with st.sidebar:
    st.header("Configuration")
    model_dir = r"C:\Users\lukea\.lmstudio\models\lmstudio-community"
    models = [os.path.join(root, file)
              for root, _, files in os.walk(model_dir)
              for file in files if file.endswith(".gguf")]
    if not models:
        st.error(f"No models found in {model_dir}")
        st.stop()
    model_path = st.selectbox("Choose LLM model", models)
    prompt_keys = list(PROMPT_TEMPLATES.keys())
    prompt_idx = st.selectbox("Prompt template", range(len(prompt_keys)), format_func=lambda i: prompt_keys[i])
    prompt_template = PROMPT_TEMPLATES[prompt_keys[prompt_idx]]
    if prompt_keys[prompt_idx] == "Custom":
        prompt_template = st.text_area("Enter your custom instruction:")
    if st.button("Save Prompt Template"):
        name = st.text_input("Template name for prompt:", key="prompt_save")
        if name and prompt_template:
            save_template(name, prompt_template, template_type="prompt")
    if st.button("Load Prompt Template"):
        loaded = load_template("prompt")
        if loaded:
            prompt_template = loaded
    save_type = st.selectbox("Save as", ["Excel", "CSV", "JSON", "Markdown"])
    merge_tables = st.checkbox("Merge all tables into one file", value=True)
    separate_excel_sheets = st.checkbox("Save each table as a separate Excel sheet (Excel only)", value=False)
    max_tokens = st.slider("Max tokens per chunk", 256, 8192, 2048, step=128)
    css_selector = st.text_input("Custom CSS selector for extraction (optional):")
    if st.button("Save CSS Selector"):
        name = st.text_input("Template name for CSS selector:", key="css_save")
        if name and css_selector:
            save_template(name, css_selector, template_type="css")
    if st.button("Load CSS Selector"):
        loaded = load_template("css")
        if loaded:
            css_selector = loaded

st.markdown("#### Enter one or more URLs (comma or newline separated) or upload a file with URLs below:")

url_input = st.text_area("Paste URLs here:")
uploaded_file = st.file_uploader("Or upload a .txt or .csv file with URLs", type=["txt", "csv"])

target_lang = st.selectbox("Select target language for translation", ["en", "es", "fr", "de", "zh", "auto"])
summarize = st.checkbox("Summarize content?")
extract_keywords = st.checkbox("Extract keywords?")
smart_extract = st.checkbox("Smart extract (AI auto-detects tables, lists, key-value pairs)?")
extract_entities = st.checkbox("Extract emails, phones, dates, and named entities?")
show_diff = st.checkbox("Show content changes since last scrape?")
extract_html_tables_opt = st.checkbox("Extract HTML tables?")
show_llm_responses = st.checkbox("Show raw LLM responses?")
show_html = st.checkbox("Show raw HTML?")
playground_prompt = st.text_area("LLM Prompt Playground (edit and try a custom prompt):", value=prompt_template)
natural_instruction = st.text_area("Natural Language Scraping Instruction (optional):", placeholder="e.g. Get all product names and prices.")
pagination = st.checkbox("Follow pagination (next page links)?", value=False)
max_pages = st.number_input("Max pages to scrape (if paginating)", min_value=1, max_value=20, value=3)

run_button = st.button("Scrape and Process")

urls = []
if uploaded_file is not None:
    content = uploaded_file.read().decode()
    if uploaded_file.name.endswith(".csv"):
        urls = pd.read_csv(io.StringIO(content)).iloc[:,0].dropna().astype(str).tolist()
    else:
        urls = [u.strip() for u in content.replace('\n', ',').split(",") if u.strip()]
elif url_input.strip():
    urls = [u.strip() for u in url_input.replace('\n', ',').split(",") if u.strip()]

if st.session_state.schedule and urls:
    st.success("Scheduled scrape running now!")
    run_button = True
    st.session_state.schedule = None

if run_button and urls:
    st.session_state.results = {}
    for url in urls:
        st.markdown(f"### Processing: {url}")
        try:
            all_tables = []
            all_entities = []
            all_responses = []
            all_htmls = []
            page_count = 0
            next_url = url
            status = "Success"
            while next_url and (not pagination or page_count < max_pages):
                page_count += 1
                st.info(f"Scraping page {page_count}: {next_url}")
                screenshot_path = os.path.join(tempfile.gettempdir(), f"screenshot_{abs(hash(next_url))}.png")

                with st.spinner("Scraping page and taking screenshot..."):
                    time.sleep(2 + (page_count - 1) * 0.5)
                    content, html = scrape_page(
                        next_url,
                        css_selector=css_selector if css_selector else None,
                        screenshot_path=screenshot_path
                    )

                all_htmls.append(html)
                lang = detect_language(content)
                st.write(f"Detected language: {lang}")
                if lang != target_lang and target_lang != "auto":
                    translate_prompt = PROMPT_TEMPLATES["Translate"].format(lang=target_lang)
                    content = gemma_llm_task(content, translate_prompt, model_path, max_tokens=max_tokens)
                changed = False
                if show_diff:
                    changed = compare_with_previous(next_url, content)
                if summarize:
                    summary = gemma_llm_task(content, PROMPT_TEMPLATES["Summarize"], model_path, max_tokens=max_tokens)
                    st.info(summary)
                if extract_keywords:
                    keywords = gemma_llm_task(content, PROMPT_TEMPLATES["Extract Keywords"], model_path, max_tokens=max_tokens)
                    st.info(keywords)
                if smart_extract:
                    smart_result = gemma_llm_task(content, PROMPT_TEMPLATES["Smart Extract"], model_path, max_tokens=max_tokens)
                    st.text_area("Smart Extract Output", smart_result, height=200)
                if extract_entities:
                    entities = extract_entities_llm(content, model_path, max_tokens=512)
                    all_entities.append(entities)
                    st.text_area("Entities Found", entities, height=150)
                if natural_instruction.strip():
                    nl_result = gemma_llm_task(content, natural_instruction, model_path, max_tokens=max_tokens)
                    st.text_area("Natural Language Extraction Output", nl_result, height=200)
                user_keywords = st.text_input(f"Enter keywords or regex patterns to search in {next_url} (comma separated):", key=next_url)
                found = search_keywords(content, [k.strip() for k in user_keywords.split(",") if k.strip()]) if user_keywords else []
                if found:
                    st.success("Found keywords/patterns: " + ", ".join(found))
                if playground_prompt.strip():
                    playground_result = gemma_llm_task(content, playground_prompt, model_path, max_tokens=max_tokens)
                    st.text_area("Playground Output", playground_result, height=200)
                instruction = prompt_template
                responses = []
                reference_columns = None
                chunks = list(chunk_text(content, max_chars=4000))
                st.write(f"Processing {len(chunks)} chunks...")
                progress = st.progress(0)
                for idx, chunk in enumerate(chunks):
                    response = gemma_llm_task(chunk, instruction, model_path, max_tokens=max_tokens)
                    responses.append(response)
                    tables = extract_markdown_tables(response, reference_columns)
                    if tables and reference_columns is None:
                        reference_columns = tables[0].columns.tolist()
                    all_tables.extend(tables)
                    progress.progress((idx + 1) / len(chunks))
                if extract_html_tables_opt:
                    html_tables = extract_html_tables(html, base_url=next_url)
                    all_tables.extend(html_tables)
                all_responses.extend(responses)
                # --- Images Found section removed here ---
                if changed and st.button("Notify me of changes", key=f"notify_{url}_{page_count}"):
                    st.success("Notification sent! (demo)")
                if pagination:
                    next_url = find_next_page_url(html, next_url)
                else:
                    next_url = None
            if all_tables:
                st.success(f"Extracted {len(all_tables)} tables.")
                merged_df = pd.concat(all_tables, ignore_index=True)
                st.dataframe(merged_df.head(100))
                st.markdown("#### Rename & Reorder Columns Before Download")
                columns = list(merged_df.columns)
                new_order = st.multiselect("Reorder columns", columns, default=columns, key=f"order_{url}")
                new_names = {}
                for col in new_order:
                    new_name = st.text_input(f"Rename column '{col}'", value=col, key=f"colmap_{col}_{url}")
                    new_names[col] = new_name
                st.session_state.column_map = new_names
                mapped_df = merged_df[new_order].rename(columns=new_names)
                plot_table(mapped_df)
                st.markdown("#### AI Data Cleaning")
                if st.button("Clean Data with AI", key=f"clean_{url}"):
                    cleaning_prompt = (
                        "Clean and normalize the following table. Remove duplicates, fix typos, and standardize formats. "
                        "Return the cleaned table in markdown format."
                    )
                    cleaned = gemma_llm_task(mapped_df.to_markdown(), cleaning_prompt, model_path)
                    st.text_area("Cleaned Table (Markdown)", cleaned, height=200)
                file_bytes, file_name = save_tables_download(
                    [mapped_df], save_type, merge=True, separate_excel_sheets=separate_excel_sheets
                )
                if file_bytes:
                    st.download_button("Download result", file_bytes, file_name)
                else:
                    st.warning("No data available for download.")
                st.markdown("#### Explainable AI Extraction")
                explain = st.checkbox("Show LLM explanation for extraction?", key=f"explain_{url}")
                if explain:
                    explanation = explain_extraction_llm(content, mapped_df.head(10).to_markdown(), model_path)
                    st.text_area("LLM Explanation", explanation, height=200)
                if webhook_url:
                    if st.button("Send results to webhook", key=f"webhook_{url}"):
                        post_to_webhook(mapped_df.to_dict(orient="records"), webhook_url)
            else:
                st.warning("No tables found.")
            if show_llm_responses:
                for i, resp in enumerate(all_responses):
                    st.text_area(f"Chunk {i+1} LLM Output", resp, height=150)
            if show_html:
                for i, html in enumerate(all_htmls):
                    st.text_area(f"HTML Page {i+1}", html[:10000], height=300)
            if os.path.exists(screenshot_path):
                st.image(screenshot_path, caption="Page Screenshot", use_column_width=True)
            st.markdown("#### AI-Assisted Selector Finder")
            user_text = st.text_input("Enter a snippet of text from the page to find its CSS selector:", key=f"selector_{url}")
            if user_text:
                selector = suggest_css_selector(html, user_text)
                if selector:
                    st.success(f"Suggested CSS selector: `{selector}`")
                else:
                    st.warning("Could not find a matching element.")
            st.markdown("#### Visual Extraction Rule Builder (Beta)")
            st.info("Click elements in the preview below to build your extraction rule (coming soon!)")
            st.session_state.dashboard.append({
                "url": url,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": status
            })
        except Exception as e:
            st.error(f"Error processing {url}: {e}")
            st.session_state.dashboard.append({
                "url": url,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "Error"
            })