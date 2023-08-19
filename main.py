"""
----------
text2video
----------
DJ Stomp 2023
No Rights Reserved
"""
import argparse
from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys


class Text2Video:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode

    def dprint(self, *args, **kwargs):
        if self.debug_mode:
            print("[DEBUG]", *args, **kwargs)

    def get_pipeline(self):
        return pipeline("text-to-video-synthesis", "damo/text-to-video-synthesis")

    def run_inference(self, prompt):
        inference = {"image": ""}
        try:
            output_video_path = self.get_pipeline()(prompt)[OutputKeys.OUTPUT_VIDEO]
            self.dprint(output_video_path)
            inference["image"] = str(output_video_path)
        except Exception as e:
            print("Ah jeez, something went wrong during the inference:\n", e)
        finally:
            return inference


def main(prompt_text=None):
    parser = argparse.ArgumentParser(
        description="Run inference to generate a video from the provided prompt"
    )
    parser.add_argument("text_input", type=str, help="Text input for the model.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    args = parser.parse_args()

    model = Text2Video(debug_mode=args.debug)
    result = model.run_inference(prompt_text if bool(prompt_text) else args.text_input)

    status, filepath = ["FAIL", "ERRNO"]
    try:
        status, filepath = [
            "PASS",
            result["image"] if os.path.isfile(result["image"]) else "ERRNO",
        ]
    except Exception as e:
        dprint(
            "\nInference failed!\n"
            + f"status: {status}, filepath: {filepath}, result['image']: {result['image']}"
        )
    finally:
        return {"status": status, "filepath": filepath}


if __name__ == "__main__":
    print(main())
