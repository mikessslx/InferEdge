Includes configurations for cAdvisor such as the metrics to scrape. Will also include
the cAdvisor binary when built by the suite. These will be transferred to target devices.

You may modify the perf_config.json file to add/remove additional performance metrics to monitor.
The available metrics vary depending on the device, but generally a good list can be found
at https://www.brendangregg.com/perf.html#Events in Section 5.