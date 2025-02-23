import logging
import sys
import pandas as pd

from whylogs.core.schema import DatasetSchema
from whylogs.experimental.core.udf_schema import udf_schema
from whylogs.viz import NotebookProfileVisualizer
from whylogs.api.logger.transient import TransientLogger


def show_metrics(result_set):
    paths = []
    for col_name, col in result_set.profile().view().get_columns().items():
        paths.extend([col_name + ":" + path for path in col.get_metric_component_paths()])
    for path in paths:
        print(path)

def llm_schema():
    _stderr = sys.stderr
    _stdout = sys.stdout
    sys.stderr = sys.stdout = None
    from langkit import llm_metrics
    schema = llm_metrics.init()
    sys.stderr = _stderr
    sys.stdout = _stdout
    return schema

def base_clean_schema(metric_name):
    schema = udf_schema()

    final_udf_spec = []
    for udf_spec in schema.multicolumn_udfs:
        if metric_name in udf_spec.udfs:
            final_udf_spec.append(udf_spec)
    schema.multicolumn_udfs = final_udf_spec
    return schema

def base_show_queries(annotated_dataset, metric_name, n, ascending):
    if ascending == None and metric_name in ["response.relevance_to_prompt"]:
        sorted_annotated_dataset = annotated_dataset.sort_values(by=[metric_name], ascending=True)
    elif ascending == None and metric_name in ["prompt.toxicity", "response.toxicity"]:
        sorted_annotated_dataset = annotated_dataset.sort_values(by=[metric_name], ascending=False)
    else:
        if ascending == None: 
            ascending = False
        sorted_annotated_dataset = annotated_dataset.sort_values(by=[metric_name], ascending=ascending)
        
    return sorted_annotated_dataset[:n][["prompt", "response", "ground_truth", metric_name]]


def show_langkit_critical_queries(dataset, metric_name, n=3, ascending=None):
    annotated_dataset, _ = base_clean_schema(metric_name).apply_udfs(dataset)
    return base_show_queries(annotated_dataset, metric_name, n, ascending)


def base_visualize_metric(dataset_or_profile, metric_name, schema, numeric):
    logging.getLogger("whylogs.viz.notebook_profile_viz").setLevel(logging.ERROR)
    if type(dataset_or_profile) == pd.DataFrame:
        prof_view = TransientLogger().log(dataset_or_profile, schema=schema).profile().view()
    else:
        prof_view = dataset_or_profile.view()

    viz = NotebookProfileVisualizer()
    viz.set_profiles(prof_view)
    if metric_name in ["prompt.has_patterns", "response.has_patterns"]:
        return viz.distribution_chart(metric_name)
    else:
        return viz.double_histogram(metric_name)
    
def visualize_langkit_metric(dataset_or_profile, metric_name, numeric=None):
    schema=base_clean_schema(metric_name)
    if numeric == None:
        if metric_name in ["prompt.has_patterns", "response.has_patterns"]:
            numeric = False
        else:
            numeric = True
    return base_visualize_metric(dataset_or_profile, metric_name, schema, numeric)
