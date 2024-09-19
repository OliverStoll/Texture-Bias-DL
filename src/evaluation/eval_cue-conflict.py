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
    categories = [
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

    def evaluate_model_texture_bias(self, model, verbose=True):
        """ Evaluate the texture bias of a model on the style-transfer-preprocessed-512 dataset. """
        texture_bias_per_category = {category: [] for category in self.categories}
        for category in self.categories:
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
                classes_softmax_dict = dict(zip(self.categories, classes_softmax))
                category_prob = classes_softmax_dict[category]
                texture_prob = classes_softmax_dict[texture]
                bias_to_texture = texture_prob / (category_prob + texture_prob)
                texture_bias_per_category[category].append(bias_to_texture)
                if verbose:
                    self.log.debug(
                        f"{category_image_name:<25} -> Texture-Bias: {bias_to_texture * 100:3.0f} %"
                        f" | {category:<8} ({category_prob:.3f}) | {texture:<8} ({texture_prob:.3f})"
                    )
        # calculate average texture bias for each category
        for category in texture_bias_per_category.keys():
            texture_bias_per_category[category] = sum(texture_bias_per_category[category]) / len(texture_bias_per_category[category])
            texture_bias_per_category[category] = round(texture_bias_per_category[category], 4)
        return texture_bias_per_category

    def evaluate_multiple_models_texture_bias(self, model_names: list, verbose=True):
        all_outputs = {}
        for idx, model_name in enumerate(model_names):
            self.log.info(f"[{idx+1}/{len(model_names)}] Evaluating cue-conflict: {model_name}")
            model = ModelCollection().get_model(model_name, CONFIG['datasets']['imagenet'],
                                                pretrained=True)
            category_tb = self.evaluate_model_texture_bias(model, verbose)
            average_texture_bias = sum(category_tb.values()) / len(category_tb)
            all_outputs[model_name] = average_texture_bias
            # log results
            self.log.info(f"Texture-Bias [{model_name}]: avg. TB: {average_texture_bias}\n")
            for category, tb in category_tb.items():
                self.log.info(f"\t{category:<10} -> {tb * 100:3.0f} %")
            category_tb['average'] = average_texture_bias

        return all_outputs


if __name__ == '__main__':
    model_names = ModelCollection().all_model_names

    texture_bias_results = CueConflictEvaluation().evaluate_multiple_models_texture_bias(
        model_names=model_names,
        verbose=True,
    )
    # save results
    with open('texture-bias-results.json', 'w') as file:
        json.dump(texture_bias_results, file)
