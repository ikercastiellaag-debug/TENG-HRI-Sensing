# TENG-HRI-Sensing

Code and data for the TENG-based tactile sensing system for human-robot interaction presented in the paper.

## Overview

This repository contains the code and data for an end-to-end tactile perception system based on triboelectric nanogenerators (TENGs), developed for the JARI social robot. The system combines spatially distributed single-electrode and contact-separation TENGs with high-impedance signal acquisition, wireless transmission, and a CNN-BiLSTM model for real-time classification of five tactile interaction states (no contact, finger stroke, finger tap, hand stroke, hand tap). Validated predictions are published as a ROS 2 topic, closing the perception-to-behavior loop with JARI's expressive architecture.

## System pipeline

TENG sensing → ESP32 acquisition and wireless transmission → signal preprocessing → CNN-BiLSTM inference → finite-state machine validation → ROS 2 publication → robot response

## Repository structure
