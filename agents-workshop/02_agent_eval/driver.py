# Databricks notebook source
# MAGIC %md
# MAGIC # Hands-On Lab: Building Agent Systems with Databricks
# MAGIC
# MAGIC ## Part 2 - Agent Evaluation
# MAGIC Now that we've created an agent, how do we evaluate its performance?
# MAGIC For the second part, we're going to create a more basic agent so we can focus on evaluation.
# MAGIC This agent will use a RAG approach to help answer questions about products using the product documentation.
# MAGIC
# MAGIC ### 2.2 Create Evaluation Dataset
# MAGIC - We've provided an example evaluation dataset - though you can also generate this [synthetically](https://www.databricks.com/blog/streamline-ai-agent-evaluation-with-new-synthetic-data-capabilities).
# MAGIC
# MAGIC ### 2.3 Run MLflow.evaluate() 
# MAGIC - MLflow will take your evaluation dataset and test your agent's responses against it
# MAGIC - LLM Judges will score the outputs and collect everything in a nice UI for review
# MAGIC
# MAGIC ### 2.4 Make Needed Improvements and re-run Evaluations
# MAGIC - Take feedback from our evaluation run and change retrieval settings
# MAGIC - Run evals again and see the improvement!

# COMMAND ----------

# MAGIC %md
# MAGIC # Driver notebook
# MAGIC
# MAGIC This was an auto-generated notebook created by an AI Playground export. We generated three notebooks in the same folder:
# MAGIC - [agent]($./agent): contains the code to build the agent.
# MAGIC - [config.yml]($./config.yml): contains the configurations.
# MAGIC - [**driver**]($./driver): logs, evaluate, registers, and deploys the agent.
# MAGIC
# MAGIC This notebook uses Mosaic AI Agent Framework ([AWS](https://docs.databricks.com/en/generative-ai/retrieval-augmented-generation.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/retrieval-augmented-generation)) to deploy the agent defined in the [agent]($./agent) notebook. The notebook does the following:
# MAGIC 1. Logs the agent to MLflow
# MAGIC 2. Evaluate the agent with Agent Evaluation
# MAGIC 3. Registers the agent to Unity Catalog
# MAGIC 4. Deploys the agent to a Model Serving endpoint
# MAGIC
# MAGIC ## Prerequisities
# MAGIC
# MAGIC - Address all `TODO`s in this notebook.
# MAGIC - Review the contents of [config.yml]($./config.yml) as it defines the tools available to your agent, the LLM endpoint, and the agent prompt.
# MAGIC - Review and run the [agent]($./agent) notebook in this folder to view the agent's code, iterate on the code, and test outputs.
# MAGIC
# MAGIC ## Next steps
# MAGIC
# MAGIC After your agent is deployed, you can chat with it in AI playground to perform additional checks, share it with SMEs in your organization for feedback, or embed it in a production application. See docs ([AWS](https://docs.databricks.com/en/generative-ai/deploy-agent.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/deploy-agent)) for details

# COMMAND ----------

# MAGIC %pip install -U -qqqq databricks-agents mlflow langchain==0.2.16 langchain-community==0.2.16 langgraph-checkpoint==1.0.12  langchain_core  langgraph==0.2.16 pydantic
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

 # %pip install -U -qqqq langchain_databricks

# COMMAND ----------

# Catalog and schema have been automatically created thanks to lab environment
#catalog_name = f"{username}_vocareum_com"
catalog_name = "marion_test"
schema_name = "email"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log the `agent` as an MLflow model
# MAGIC Log the agent as code from the [agent]($./agent) notebook. See [MLflow - Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code).

# COMMAND ----------

# Log the model to MLflow
import os
import mlflow

input_example = {
    "messages": [
        {
            "role": "user",
            "content": "Can you draft an marketing email for customer David SAnchez based on the items found in his browsing history ?"
        }
    ]
}

with mlflow.start_run():
    logged_agent_info = mlflow.langchain.log_model(
        lc_model=os.path.join(
            os.getcwd(),
            'agent',
        ),
        pip_requirements=[
            "langchain==0.2.16",
            "langchain-community==0.2.16",
            "langgraph-checkpoint==1.0.12",
            "langgraph==0.2.16",
            "pydantic",
            #"langchain_databricks", # used for the retriever tool
        ],
        model_config="config.yml",
        artifact_path='agent',
        input_example=input_example,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the agent with [Agent Evaluation](https://docs.databricks.com/generative-ai/agent-evaluation/index.html)
# MAGIC
# MAGIC You can edit the requests or expected responses in your evaluation dataset and run evaluation as you iterate your agent, leveraging mlflow to track the computed quality metrics.

# COMMAND ----------

import pandas as pd

data = {
    "request": [
        "What product recommendations could we make for customer David Sanchez based on his latest order? ",
        "What product recommendations could we make for customer David Sanchez based on his wishlisted items?"
    ],
    "expected_facts": [
        [
            "Products the most similar to the ones in the customer's cart are: 1. Product 1, 2. Product 2"
            "Products the customer has previously purchased are: product 3, very similar in texture and color to product 4"
        ]
    ]
}

eval_dataset = pd.DataFrame(data)

# COMMAND ----------

# DBTITLE 1,Run MLflow Evaluations!
import mlflow
import pandas as pd

with mlflow.start_run(run_id=logged_agent_info.run_id):
    eval_results = mlflow.evaluate(
        f"runs:/{logged_agent_info.run_id}/agent",  # replace `chain` with artifact_path that you used when calling log_model.
        data=eval_dataset,  # Your evaluation dataset
        model_type="databricks-agent",  # Enable Mosaic AI Agent Evaluation
    )

# Review the evaluation results in the MLFLow UI (see console output), or access them in place:
display(eval_results.tables['eval_results'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lets go back to the [agent]($./agent) notebook and change our retriever to 1.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC
# MAGIC Update the `catalog`, `schema`, and `model_name` below to register the MLflow model to Unity Catalog.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import os

mlflow.set_registry_uri("databricks-uc")

# Use the workspace client to retrieve information about the current user
w = WorkspaceClient()
user_email = w.current_user.me().display_name
username = user_email.split("@")[0]



# TODO: define the catalog, schema, and model name for your UC model
model_name = "product_agent"
UC_MODEL_NAME = f"{catalog_name}.{schema_name}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME)

# COMMAND ----------

from IPython.display import display, HTML

# Retrieve the Databricks host URL
workspace_url = spark.conf.get('spark.databricks.workspaceUrl')

# Create HTML link to created agent
html_link = f'<a href="https://{workspace_url}/explore/data/models/{catalog_name}/{schema_name}/product_agent" target="_blank">Go to Unity Catalog to see Registered Agent</a>'
display(HTML(html_link))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the agent
# MAGIC
# MAGIC ##### Note: This is disabled for lab users but will work on your own workspace

# COMMAND ----------

from databricks import agents

# Deploy the model to the review app and a model serving endpoint
agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, tags = {"endpointSource": "playground"})
