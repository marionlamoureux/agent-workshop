# Databricks notebook source
# MAGIC %md
# MAGIC # Hands-On Lab: Building Agent Systems with Databricks
# MAGIC
# MAGIC ## Part 1 - Architect Your First Agent
# MAGIC This first agent will follow the workflow of a customer service representative to illustrate the various agent capabilites. 
# MAGIC We'll focus around processing product returns as this gives us a tangible set of steps to follow.
# MAGIC
# MAGIC ### 1.1 Build Simple Tools
# MAGIC - **SQL Functions**: Create queries that access data critical to steps in the customer service workflow for processing a return.
# MAGIC - **Simple Python Function**: Create and register a Python function to overcome some common limitations of language models.
# MAGIC
# MAGIC ### 1.2 Integrate with an LLM [AI Playground]
# MAGIC - Combine the tools you created with a Language Model (LLM) in the AI Playground.
# MAGIC
# MAGIC ### 1.3 Test the Agent [AI Playground]
# MAGIC - Ask the agent a question and observe the response.
# MAGIC - Dive deeper into the agent’s performance by exploring MLflow traces.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Library Installs
# MAGIC %pip install -qqqq -U -r requirements.txt
# MAGIC # Restart to load the packages into the Python environment
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import urllib
import os
from databricks.sdk.service import catalog
from databricks.sdk import WorkspaceClient
import yaml
import mlflow

# Allows us to reference these values directly in the SQL/Python function creation
dbutils.widgets.text("catalog_name", defaultValue=catalog_name, label="Catalog Name")
dbutils.widgets.text("schema_name", defaultValue=schema_name, label="Schema Name")
catalog_name = "marion_test"
schema_name = "email"
volume="data"


files = ['browsing_history', 'customers', 'email_logs', 'product_catalog', 'purchases']
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog_name}.{schema_name}.{volume}")
base_url = 'https://raw.githubusercontent.com/marionlamoureux/agent-workshop/June2025/agents-workshop/data/'

for d in files:
  file_name = d+'.csv'
  url = base_url+file_name
  urllib.request.urlretrieve(url, f'/Volumes/{catalog_name}/{schema_name}/{volume}/{file_name}')
  df_csv = spark.read.csv(f'/Volumes/{catalog_name}/{schema_name}/{volume}/{file_name}',
    header=True,
    sep=",")
  df_csv.write.option("mergeSchema", "true").mode("append").saveAsTable(f'{catalog_name}.{schema_name}.{d}')

# COMMAND ----------

# MAGIC %md
# MAGIC # Customer Service Return Processing Workflow
# MAGIC
# MAGIC Below is a structured outline of the **key steps** a customer service agent would typically follow when **processing a return**. This workflow ensures consistency and clarity across your support team.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Define our retriever tool. 
# MAGIC Vector Search: We've created a Vector Search endpoint that can be queried to find related documentation about a specific product. Create Retriever Function: Define some properties about our retriever and package it so it can be called by our LLM. Note: You can also change the system prompt as we defined in the playground within the config.yml file
# MAGIC
# MAGIC Let's build a vector search index out of the product catalog that contains a text column "description"

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE ${catalog_name}.${schema_name}.product_catalog SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

vector_endpoint_name = f'vs_endpoint_product_{schema_name}5'
index_name = f'{catalog_name}.{schema_name}.product_catalog_index'
uc_function = f'{catalog_name}.{schema_name}.*'

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
# The following line automatically generates a PAT Token for authentication
client = VectorSearchClient()

# The following line uses the service principal token for authentication
# client = VectorSearchClient(service_principal_client_id=<CLIENT_ID>,service_principal_client_secret=<CLIENT_SECRET>)

client.create_endpoint(
    name=vector_endpoint_name,
    endpoint_type="STANDARD")

index = client.create_delta_sync_index(
  endpoint_name=vector_endpoint_name,
  source_table_name=f'{catalog_name}.{schema_name}.product_catalog',
  index_name=index_name,
  pipeline_type="TRIGGERED",
  primary_key="item_id",
  embedding_source_column="description",
  embedding_model_endpoint_name="databricks-gte-large-en"
)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION ${catalog_name}.${schema_name}.product_vector_search (
# MAGIC   -- The agent uses this comment to determine how to generate the query string parameter.
# MAGIC   query STRING
# MAGIC   COMMENT 'The query string for searching the product catalog.'
# MAGIC ) 
# MAGIC RETURNS TABLE
# MAGIC -- The agent uses this comment to determine when to call this tool. It describes the types of documents and information contained within the index.
# MAGIC COMMENT 'Executes a search on the product catalog to retrieve text documents most relevant to the input query.'
# MAGIC RETURN
# MAGIC SELECT
# MAGIC   description,
# MAGIC   item_id
# MAGIC FROM
# MAGIC   vector_search(
# MAGIC     -- Specify your Vector Search index name here
# MAGIC     index => 'marion_test.agents.product_catalog_index',
# MAGIC     query => query,
# MAGIC     num_results => 5
# MAGIC   )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get the Latest Return in the Processing Queue
# MAGIC - **Action**: Identify and retrieve the most recent return request from the ticketing or returns system.  
# MAGIC - **Why**: Ensures you’re working on the most urgent or next-in-line customer issue.
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Get the Latest Order for a customer
# MAGIC %sql
# MAGIC -- First lets make sure it doesnt already exist
# MAGIC DROP FUNCTION IF EXISTS ${catalog_name}.${schema_name}.return_last_order;
# MAGIC
# MAGIC -- Now we create our first function. This takes in one parameter and returns the most recent interaction for a given customer.
# MAGIC CREATE OR REPLACE FUNCTION
# MAGIC ${catalog_name}.${schema_name}.return_last_order(customer_name STRING)
# MAGIC RETURNS TABLE(purchase_date DATE, item_id STRING, customer_id STRING, `name` STRING)
# MAGIC COMMENT 'Returns the most recent purchase from our luxury fashion eshop for the customer'
# MAGIC RETURN
# MAGIC SELECT p.purchase_date, p.item_id, p.customer_id, c.name
# MAGIC FROM ${catalog_name}.${schema_name}.purchases  p
# MAGIC JOIN ${catalog_name}.${schema_name}.customers c 
# MAGIC ON p.customer_id = c.customer_id
# MAGIC WHERE c.`name` = initcap(customer_name)
# MAGIC ORDER BY p.purchase_date DESC
# MAGIC LIMIT 1;

# COMMAND ----------

# DBTITLE 1,Create a function registered to Unity Catalog
# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION ${catalog_name}.${schema_name}.return_browsing_history(customer_name STRING)
# MAGIC RETURNS TABLE (customer_id STRING, item_id STRING, action STRING)
# MAGIC COMMENT 'Returns the most recent browsing history for the customer on the luxury fashion eshop'
# MAGIC LANGUAGE SQL
# MAGIC RETURN 
# MAGIC SELECT b.customer_id, b.item_id, b.action
# MAGIC FROM ${catalog_name}.${schema_name}.browsing_history b
# MAGIC JOIN ${catalog_name}.${schema_name}.customers c 
# MAGIC ON b.customer_id = c.customer_id
# MAGIC WHERE c.name = initcap(customer_name);

# COMMAND ----------

# DBTITLE 1,Test function call to retrieve latest order
# MAGIC %sql
# MAGIC select * from ${catalog_name}.${schema_name}.return_browsing_history('david Sanchez')

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 2. Retrieve Email logs
# MAGIC - **Action**: Access the lastest emails sent to a customer
# MAGIC - **Why**: Verifying against anti-spamming policies the frequency of the emails and check conversion rate
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Create function to retrieve email log
# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION ${catalog_name}.${schema_name}.return_email_log(customer_name STRING)
# MAGIC RETURNS TABLE(customer_id STRING, subject STRING,	sent_date DATE,	opened BOOLEAN,	clicked BOOLEAN)
# MAGIC COMMENT 'This takes the name of a customer as an input and returns the email log'
# MAGIC LANGUAGE SQL
# MAGIC RETURN
# MAGIC SELECT e.customer_id, e.subject, e.sent_date, e.opened, e.clicked
# MAGIC FROM ${catalog_name}.${schema_name}.email_logs e
# MAGIC JOIN ${catalog_name}.${schema_name}.customers c ON e.customer_id = c.customer_id
# MAGIC WHERE c.name = initcap(customer_name);

# COMMAND ----------

# DBTITLE 1,Test function to retrieve email log
# MAGIC %sql
# MAGIC select * from ${catalog_name}.${schema_name}.return_email_log("David sanchez")

# COMMAND ----------

# MAGIC %md
# MAGIC # Get the last order
# MAGIC Get the customer's last order on the eshop

# COMMAND ----------

# DBTITLE 1,Get the last order
# MAGIC %sql
# MAGIC -- First lets make sure it doesnt already exist
# MAGIC DROP FUNCTION IF EXISTS ${catalog_name}.${schema_name}.return_last_order;
# MAGIC
# MAGIC -- Now we create our first function. This takes in one parameter and returns the most recent interaction for a given customer.
# MAGIC CREATE OR REPLACE FUNCTION
# MAGIC ${catalog_name}.${schema_name}.return_last_order(customer_name STRING)
# MAGIC RETURNS TABLE(purchase_date DATE, item_id STRING, customer_id STRING, `name` STRING)
# MAGIC COMMENT 'Returns the most recent purchase for the customer for the luxury good eshop'
# MAGIC RETURN
# MAGIC SELECT p.purchase_date, p.item_id, p.customer_id, c.name
# MAGIC FROM ${catalog_name}.${schema_name}.purchases  p
# MAGIC JOIN ${catalog_name}.${schema_name}.customers c 
# MAGIC ON p.customer_id = c.customer_id
# MAGIC WHERE c.`name` = initcap(customer_name)
# MAGIC ORDER BY p.purchase_date DESC
# MAGIC LIMIT 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from ${catalog_name}.${schema_name}.return_last_order('David Sanchez')

# COMMAND ----------

# MAGIC %md
# MAGIC # Retrieve the user's browsing history. 
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION ${catalog_name}.${schema_name}.return_browsing_history(customer_name STRING)
# MAGIC RETURNS TABLE (customer_id STRING, item_id STRING, action STRING)
# MAGIC COMMENT 'Returns the most recent browsing history for the customer'
# MAGIC LANGUAGE SQL
# MAGIC RETURN 
# MAGIC SELECT b.customer_id, b.item_id, b.action
# MAGIC FROM ${catalog_name}.${schema_name}.browsing_history b
# MAGIC JOIN ${catalog_name}.${schema_name}.customers c 
# MAGIC ON b.customer_id = c.customer_id
# MAGIC WHERE c.name = initcap(customer_name);

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from ${catalog_name}.${schema_name}.return_browsing_history('david Sanchez')

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## Give the LLM a Python Function to Know Today’s Date
# MAGIC - **Action**: Provide a **Python function** that can supply the Large Language Model (LLM) with the current date.  
# MAGIC - **Why**: Automating date retrieval helps in scheduling pickups, refund timelines, and communication deadlines.
# MAGIC
# MAGIC ###### Note: There is also a function registered in System.ai.python_exec that will let your LLM run generated code in a sandboxed environment
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Very simple Python function
def get_todays_date() -> str:
    """
    Returns today's date in 'YYYY-MM-DD' format.

    Returns:
        str: Today's date in 'YYYY-MM-DD' format.
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")

# COMMAND ----------

# DBTITLE 1,Test python function
today = get_todays_date()
today

# COMMAND ----------

# DBTITLE 1,Register python function to Unity Catalog
from unitycatalog.ai.core.databricks import DatabricksFunctionClient

client = DatabricksFunctionClient()

# this will deploy the tool to UC, automatically setting the metadata in UC based on the tool's docstring & typing hints
python_tool_uc_info = client.create_python_function(func=get_todays_date, catalog=catalog_name, schema=schema_name, replace=True)

# the tool will deploy to a function in UC called `{catalog}.{schema}.{func}` where {func} is the name of the function
# Print the deployed Unity Catalog function name
print(f"Deployed Unity Catalog function name: {python_tool_uc_info.full_name}")

# COMMAND ----------

# DBTITLE 1,Let's take a look at our created functions
from IPython.display import display, HTML

# Retrieve the Databricks host URL
workspace_url = spark.conf.get('spark.databricks.workspaceUrl')

# Create HTML link to created functions
html_link = f'<a href="https://{workspace_url}/explore/data/functions/{catalog_name}/{schema_name}/get_todays_date" target="_blank">Go to Unity Catalog to see Registered Functions</a>'
display(HTML(html_link))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Now lets go over to the AI Playground to see how we can use these functions and assemble our first Agent!
# MAGIC
# MAGIC ### The AI Playground can be found on the left navigation bar under 'Machine Learning' or you can use the link created below

# COMMAND ----------

# DBTITLE 1,Create link to AI Playground
# Create HTML link to AI Playground
html_link = f'<a href="https://{workspace_url}/ml/playground" target="_blank">Go to AI Playground</a>'
display(HTML(html_link))
