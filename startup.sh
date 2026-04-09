#!/bin/bash

cat .env.examle > .env
docker-compose up --build