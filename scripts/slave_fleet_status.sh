aws ec2 describe-fleets --filters '{"Name": "activity-status", "Values": ["fulfilled"]}' --profile agilemobile
aws ec2 describe-fleets --filters '{"Name": "fleet-state", "Values": ["active"]}' --profile agilemobile