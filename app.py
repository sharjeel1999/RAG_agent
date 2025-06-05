import gradio as gr


block = gr.Blocks(css = ".gradio-container {background-color: lightgray}")

with block:
    with gr.Row():
        gr.Markdown(
            "<h3><center>Chat-Your-Data (State-of-the-Union)</center></h3>")

        openai_api_key_textbox = gr.Textbox(
            placeholder = "Paste your OpenAI API key (sk-...)",
            show_label = False,
            lines = 1,
            type = "password",
        )

    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label = "What's your question?",
            placeholder = "Ask questions",
            lines = 1,
        )
        submit = gr.Button(value = "Send", variant = "secondary")#.style(full_width=False)


block.launch(debug = True)
