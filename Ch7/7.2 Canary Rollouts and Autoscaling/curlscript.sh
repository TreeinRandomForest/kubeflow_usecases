#!/bin/bash
  for i in {1..100}; do
  curl -v -H "Host: pytorchfraud.kubeflow-user-example-com.example.com" http://localhost:8080/v1/models/pytorchfraud:predict -d @./input.json
  sleep 2
done

