#!/bin/bash
# スクリプトを配置
gcloud storage cp gs://mixi-ml-workbench-notebook-utils/notebook-auto-shutdown.sh /tmp/notebook-auto-shutdown.sh
chmod +x /tmp/notebook-auto-shutdown.sh

# 10分ごとに実行
(
    crontab -l 2>/dev/null
    echo "*/10 * * * * /tmp/notebook-auto-shutdown.sh"
) | crontab -
