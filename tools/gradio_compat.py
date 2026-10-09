"""Keep the desktop Gradio 3 UI and Colab Gradio 6 on the same code path."""

import inspect

import gradio as gr


def queue_app(app):
    options = {"max_size": 1022}
    parameter = (
        "default_concurrency_limit"
        if "default_concurrency_limit" in inspect.signature(app.queue).parameters
        else "concurrency_count"
    )
    options[parameter] = 511
    return app.queue(**options)


def blocks_css_options(css):
    return {} if int(gr.__version__.split(".")[0]) >= 6 else {"css": css}


def launch_css_options(css):
    return {"css": css} if int(gr.__version__.split(".")[0]) >= 6 else {}


def audio_upload_options():
    if "sources" in inspect.signature(gr.Audio).parameters:
        return {"sources": ["upload"]}
    return {"source": "upload"}
