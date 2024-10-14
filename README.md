# RHEL AI POC 

This repository contains the code and documentation for creating RHEL AI Proofs of Concept (POCs).  This aims to be straightforward and comply with the available documentation for [RHEL AI](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux_ai/) and [InstructLab](https://github.com/instructlab/instructlab/blob/main/README.md). The goal is to provide a simple and easy to follow guide for creating successful and reproducible POCs that can be used to demonstrate the capabilities of RHEL AI.

Much of what is here, are some nuance and gotchas that may not be covered in the main documentation.  There are some included scripts and automation, but many of these are anticipated to be removed as the features are integrated into the main RHEL AI product.


## Table of Contents
<!-- TOC -->
* [RHEL AI POC](#rhel-ai-poc-)
  * [Table of Contents](#table-of-contents)
  * [Suggested documentation](#suggested-documentation)
  * [Prerequisites](#prerequisites)
  * [Installation and Configuration](#installation-and-configuration)
    * [Bare Metal](#bare-metal)
    * [IBM Cloud](#ibm-cloud)
    * [AWS](#aws)
      * [Creating an AWS AMI](#creating-an-aws-ami)
      * [Launching an AWS Instance](#launching-an-aws-instance)
  * [Document Collection](#document-collection)
    * [PDF](#pdf-)
    * [qna.yaml](#qnayaml)
    * [Evaluation Questions](#evaluation-questions)
  * [Data Preparation](#data-preparation)
    * [PDF to Markdown Conversion](#pdf-to-markdown-conversion)
  * [Synthetic Data Generation](#synthetic-data-generation)
    * [ilab generate](#ilab-generate)
  * [Model Training](#model-training)
    * [ilab train](#ilab-train)
  * [Deployment and Testing](#deployment-and-testing)
    * [Saving the model to S3 storage](#saving-the-model-to-s3-storage)
    * [Serving the model](#serving-the-model)
      * [ilab serve](#ilab-serve)
      * [OpenShift AI Serving](#openshift-ai-serving)
  * [Evaluation](#evaluation)
    * [Testing the model](#testing-the-model)
    * [RAG with Anything LLM](#rag-with-anything-llm)
    * [RAG testing with InstructLab](#rag-testing-with-instructlab)
<!-- TOC -->

## Suggested documentation
* [RHEL AI](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux_ai/) 
* [InstructLab](https://github.com/instructlab/instructlab/blob/main/README.md)
## Prerequisites


## Installation and Configuration

### Bare Metal
TBD

### IBM Cloud
TBD

### AWS


#### Creating an AWS AMI


#### Launching an AWS Instance



## Document Collection

### PDF 

### qna.yaml

### Evaluation Questions


## Data Preparation

### PDF to Markdown Conversion



## Synthetic Data Generation

### ilab generate



## Model Training

### ilab train




## Deployment and Testing

### Saving the model to S3 storage

### Serving the model

#### ilab serve

#### OpenShift AI Serving



## Evaluation

### Testing the model

### RAG with Anything LLM

### RAG testing with InstructLab

