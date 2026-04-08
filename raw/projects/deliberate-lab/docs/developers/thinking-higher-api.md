---
layout: default
title: Thinking Higher API Guide
---

# Thinking Higher: API Integration Guide

This guide explains how to programmatically create and run the "Thinking Higher: I Thought It Worked" workplace simulation using the Deliberate Lab API.

For general API reference and authentication details, see the [API Reference](/developers/api).

## Overview

The Thinking Higher experiment involves:
1. Creating an experiment from the `thinking_higher` template.
2. Creating a cohort (which automatically enrolls the 4 AI mediators).
3. Adding a participant to the cohort.
4. Fetching the resulting transcripts and assessment data.

## 1. Create the Experiment

First, create a new experiment instance using the Thinking Higher template ID.

```bash
curl -X POST https://api.deliberatelab.org/v1/experiments \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "templateId": "thinking_higher",
    "metadata": {
      "name": "My Thinking Higher Run",
      "description": "Programmatic API run for workplace sim"
    }
  }'
```

**Response:**
```json
{
  "experimentId": "exp-12345"
}
```

## 2. Create a Cohort

Once the experiment is created, initialize a cohort. The 4 AI personas (Marcus, Alex, Sarah, Evaluator) are marked as `isDefaultAddToCohort: true` in the template, so they will be automatically instantiated and added to this cohort.

```bash
curl -X POST https://api.deliberatelab.org/v1/experiments/exp-12345/cohorts \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "metadata": {
      "name": "Cohort 1"
    }
  }'
```

**Response:**
```json
{
  "cohortId": "cohort-67890"
}
```

## 3. Add a Participant

Now, add a human participant (or an AI agent) to the cohort. This will generate a unique URL for the participant to join the simulation.

```bash
curl -X POST https://api.deliberatelab.org/v1/experiments/exp-12345/cohorts/cohort-67890/participants \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "profile": {
      "name": "Jane Doe",
      "avatar": "👩‍💻"
    }
  }'
```

**Response:**
```json
{
  "participantId": "part-abcde",
  "joinUrl": "https://deliberatelab.org/e/exp-12345/join?p=part-abcde"
}
```
*Note: Share the `joinUrl` with your participant. They will go through the 8 stages (TOS, Profile, Group Standup, 3x 1-on-1s, Assessment, Survey).*

## 4. Fetch Transcripts & Data

After the participant completes the simulation, you can fetch their chat transcripts and the final ELIPSS assessment.

```bash
curl -X GET "https://api.deliberatelab.org/v1/experiments/exp-12345/cohorts/cohort-67890/transcripts?participantId=part-abcde" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response:**
The response will include the chat history across all stages, including the final assessment from the Evaluator:

```json
{
  "transcripts": [
    {
      "stageId": "assessment-chat",
      "messages": [
        {
          "sender": "Evaluator",
          "text": "**Overall Assessment**\n\n1. Critical Thinking - 4/5\n..."
        }
      ]
    }
  ]
}
```
