import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from timm.models import create_model

from models import ModelFactory
from modelvshuman import Plot, Evaluate
from modelvshuman import constants as c
from modelvshuman.plotting.colors import *
from modelvshuman.plotting.decision_makers import DecisionMaker


def plotting_definition_template(df):
    """Decision makers to compare a few models with human observers."""
    decision_makers = []
    for model in MODELS:
        decision_makers.append(DecisionMaker(name_pattern=model['model_name'],
                                             color=model['color'],
                                             marker="o", df=df,
                                             plotting_name=model['model_name']))
    decision_makers.append(DecisionMaker(name_pattern="subject-*",
                           color=rgb(165, 30, 55), marker="D", df=df,
                           plotting_name="humans"))
    return decision_makers


def run_evaluation(models, datasets=None):
    # datasets = c.DEFAULT_DATASETS  # or e.g. ["cue-conflict", "uniform-noise"]
    datasets = datasets or c.DEFAULT_DATASETS  # MINE
    params = {"batch_size": 64, "print_predictions": True, "num_workers": 20}
    Evaluate()(models, datasets, **params)


def run_plotting(plot_types=None):
    plot_types = plot_types or c.DEFAULT_PLOT_TYPES
    plotting_def = plotting_definition_template
    figure_dirname = "example-figures/"
    Plot(
        plot_types=plot_types,
        plotting_definition=plotting_def,
        figure_directory_name=figure_dirname,  # fixed in code
        crop_PDFs=False,
    )


def _get_models(model_names):
    default_colors = [blue1, green1, grey1, orange1, purple1, red1, metallic, brown1, black, blue3]
    models = []
    for i, model_name in enumerate(model_names):
        models.append({
            'model_name': model_name,
            'model': create_model(model_name, pretrained=True),
            'framework': 'pytorch',
            'color': default_colors[i],
        })
    return models


if __name__ == "__main__":

    MODELS = _get_models(model_names=ModelFactory.transformer_models.values())

    run_evaluation(MODELS, datasets=['cue-conflict'])
    run_plotting(plot_types=['shape-bias'])
