import os
import json
from PIL import Image
from util_code.example_image import to_tensor

from models import ModelCollection
from utils.config import CONFIG
from utils.logger import create_logger
from probabilities_to_decision import ImageNetProbabilitiesTo16ClassesMapping


class CueConflictEvaluation:
    log = create_logger("Cue-Conflict Evaluation")
    stimuli_path = '/home/olivers/colab-master-thesis/src/evaluation/cue-conflict/texture-vs-shape/stimuli/style-transfer-preprocessed-512'
    stimuli_categories = [
        "airplane",
        "bear",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "car",
        "cat",
        "chair",
        "clock",
        "dog",
        "elephant",
        "keyboard",
        "knife",
        "oven",
        "truck"
    ]
    mapping = ImageNetProbabilitiesTo16ClassesMapping()  # function provided by the paper

    def evaluate_model_texture_bias(self, model: str):
        """ Evaluate the texture bias of a model on the style-transfer-preprocessed-512 dataset. """
        texture_bias_outputs = []
        for category in self.stimuli_categories:
            category_path = f"{self.stimuli_path}/{category}"
            category_image_names = os.listdir(category_path)
            for category_image_name in category_image_names:
                texture_name_raw = category_image_name.split('-')[1].replace('.png', '')
                texture = ''.join([i for i in texture_name_raw if not i.isdigit()])
                example_image = Image.open(f"{category_path}/{category_image_name}")
                example_tensor = to_tensor(example_image)
                example_tensor = example_tensor.unsqueeze(0)
                output = model(example_tensor)
                output = output.squeeze(0)
                softmax_output = output.softmax(dim=0).detach().numpy()
                output = output.detach().numpy()
                predicted_category = self.mapping.probabilities_to_decision(softmax_output)

                # check if predicted is either category or texture, else skip (according to the paper)
                if not (predicted_category == category or predicted_category == texture):
                    continue

                classes_softmax = self.mapping.softmax_of_16_classes(raw_probabilities=output)
                classes_softmax_dict = dict(zip(self.stimuli_categories, classes_softmax))
                category_prob = classes_softmax_dict[category]
                texture_prob = classes_softmax_dict[texture]
                bias_to_texture = texture_prob / (category_prob + texture_prob)
                texture_bias_outputs.append(bias_to_texture)
                self.log.debug(
                    f"{category_image_name:<25} -> Texture-Bias: {bias_to_texture * 100:.0f} %"
                    f" | {category} ({category_prob:.3f}) | {texture:<8} ({texture_prob:.3f})"
                )

        return texture_bias_outputs

    def evaluate_multiple_models_texture_bias(self, model_names: list):
        all_outputs = {}
        for idx, model_name in enumerate(model_names):
            self.log.info(f"[{idx+1}/{len(model_names)}] Evaluating cue-conflict TB for model: {model_name}")
            model = ModelCollection().get_model(model_name, CONFIG['datasets']['imagenet'],
                                                pretrained=True)
            texture_bias_outputs = self.evaluate_model_texture_bias(model)
            average_texture_bias = sum(texture_bias_outputs) / len(texture_bias_outputs)
            all_outputs[model_name] = average_texture_bias
            self.log.info(f"Texture-Bias [{model_name}]: {average_texture_bias * 100:.0f}%")

        return all_outputs


if __name__ == '__main__':
    model_names = ModelCollection().all_model_names
    texture_bias_results = CueConflictEvaluation().evaluate_multiple_models_texture_bias(model_names)
    # save results
    with open('texture-bias-results.json', 'w') as file:
        json.dump(texture_bias_results, file)
