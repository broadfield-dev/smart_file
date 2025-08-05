import gradio as gr
import pandas as pd
from pathlib import Path
import os

# Import the library we just created!
from smartfile import SemanticExplorer, utils

# --- Global Instance of our Explorer ---
file_explorer = SemanticExplorer()

# --- UI Definition ---
with gr.Blocks(theme=gr.themes.Soft(), title="Smart File Explorer Demo") as demo:
    gr.Markdown("# 🚀 Smart File Explorer Demo")
    gr.Markdown("A demonstration of the `Smart File` library for filesystem exploration and semantic search.")

    initial_path = Path.cwd().anchor
    current_dir_state = gr.State(value=initial_path)

    with gr.Tabs():
        # --- TAB 1: FILE EXPLORER ---
        with gr.TabItem("📂 File Explorer"):
            # ... (Layout is identical to before) ...
            with gr.Row():
                with gr.Column(scale=1, min_width=250):
                    gr.Markdown("### Navigation")
                    up_button = gr.Button("⬆️ Go Up")
                    home_button = gr.Button("🏠 Go to App Home")
                    root_button = gr.Button("̸ Go to Root")
                    gr.Markdown("### Go to Path:")
                    path_input = gr.Textbox(label="Enter path and press Enter", value=initial_path, interactive=True)
                with gr.Column(scale=3):
                    current_path_display = gr.Label(label="Current Directory")
                    file_list_df = gr.DataFrame(headers=["Name", "Type", "Size (bytes)", "Modified", "Permissions"], interactive=True, row_count=(15, "dynamic"))
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### File Content Viewer")
                    selected_file_path = gr.Textbox(label="Selected File Path", interactive=False)
                    file_content_display = gr.Code(label="Content", language=None, lines=20, interactive=False)

        # --- TAB 2: SEMANTIC SEARCH ---
        with gr.TabItem("🔎 Semantic Search"):
            # ... (Layout is identical to before) ...
            gr.Markdown("### Persistent Vector Search")
            gr.Markdown("The index is stored on disk and loaded on startup. Re-building the index will update or add new files.")
            with gr.Row():
                index_path_input = gr.Textbox(label="Directory to Index", value=os.getcwd(), interactive=True)
                with gr.Column():
                    build_index_button = gr.Button("Build / Update Index", variant="primary", visible=True)
                    stop_button = gr.Button("Stop Building", variant="stop", visible=False)
            with gr.Row():
                clear_index_button = gr.Button("Clear Entire Index", variant="stop")
            index_status_label = gr.Label(label="Index Status")
            with gr.Row():
                search_query_input = gr.Textbox(label="Search Query", placeholder="e.g., 'a function to login a user' or 'test cases for the API'", interactive=True, scale=4)
            gr.Markdown("### Search Results")
            search_results_df = gr.DataFrame(headers=["Similarity", "Path", "Type", "Size (Bytes)", "Modified"], interactive=False)

        # --- TABS 3 & 4: SYSTEM INFO ---
        with gr.TabItem("ℹ️ System & Dependencies"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Python Packages (`pip freeze`)")
                    refresh_pip_button = gr.Button("Refresh Pip List")
                    pip_list_display = gr.Code(language="shell", lines=20)
                with gr.Column():
                    gr.Markdown("### Disk Usage")
                    refresh_sysinfo_button = gr.Button("Refresh System Info")
                    sysinfo_display = gr.Code(language="shell", lines=20)


    # --- Event Handlers (Now they call the library methods) ---
    def update_file_list(path_str):
        df, label = utils.get_directory_contents(path_str)
        return df, label, path_str
    
    # ... (All other file browser handlers are the same, but they now call `utils` functions) ...
    demo.load(update_file_list, [current_dir_state], [file_list_df, current_path_display, current_dir_state])
    path_input.submit(update_file_list, [path_input], [file_list_df, current_path_display, current_dir_state])
    def handle_row_select(evt: gr.SelectData, df: pd.DataFrame, current_dir: str):
        if evt.index is None: return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        item_path = os.path.join(current_dir, df.iloc[evt.index[0]]["Name"])
        if df.iloc[evt.index[0]]["Type"] == "📁 Folder":
            new_df, new_label = utils.get_directory_contents(item_path)
            return new_df, new_label, "# This is a directory.", "", item_path
        else:
            content, filepath = utils.get_file_content(item_path)
            return gr.update(), gr.update(), content, filepath, gr.update()
    file_list_df.select(handle_row_select, [file_list_df, current_dir_state], [file_list_df, current_path_display, file_content_display, selected_file_path, current_dir_state])
    up_button.click(lambda d: update_file_list(str(Path(d).parent)), [current_dir_state], [file_list_df, current_path_display, current_dir_state])
    home_button.click(lambda: update_file_list(os.getcwd()), [], [file_list_df, current_path_display, current_dir_state])
    root_button.click(lambda: update_file_list(Path.cwd().anchor), [], [file_list_df, current_path_display, current_dir_state])
    refresh_pip_button.click(utils.get_pip_freeze, [], pip_list_display)
    refresh_sysinfo_button.click(utils.get_disk_usage, [], sysinfo_display)

    # --- Search Handlers ---
    def do_build_index(path, progress=gr.Progress(track_tqdm=True)):
        # This is a wrapper that calls the generator from our library
        for status in file_explorer.index_directory(path):
            progress(0, desc=status) # Use progress to show status
            yield status # Yield status to update the label

    def do_search(query):
        results = file_explorer.search(query)
        # Format for DataFrame
        df_data = [{
            "Similarity": f"{r['similarity']:.3f}",
            "Path": r['path'],
            "Type": r['type'],
            "Size (Bytes)": r['size'] if r['size'] is not None else "",
            "Modified": r['modified'].strftime('%Y-%m-%d %H:%M')
        } for r in results]
        return pd.DataFrame(df_data)

    def do_clear_index():
        count = file_explorer.clear_index()
        return f"Successfully cleared {count} items from the index."
    
    # --- UI Wiring ---
    demo.load(file_explorer.get_status, outputs=[index_status_label])

    def start_indexing(): return gr.update(visible=False), gr.update(visible=True)
    def stop_indexing(): file_explorer.cancel_indexing(); return gr.update(visible=True), gr.update(visible=False)
    def finish_indexing(): return gr.update(visible=True), gr.update(visible=False)

    click_event = build_index_button.click(start_indexing, outputs=[build_index_button, stop_button]) \
        .then(do_build_index, inputs=[index_path_input], outputs=[index_status_label]) \
        .then(finish_indexing, outputs=[build_index_button, stop_button])
    
    stop_button.click(stop_indexing, outputs=[build_index_button, stop_button], cancels=[click_event])
    clear_index_button.click(do_clear_index, outputs=[index_status_label])
    search_query_input.submit(do_search, inputs=[search_query_input], outputs=[search_results_df])

if __name__ == "__main__":
    demo.launch()
