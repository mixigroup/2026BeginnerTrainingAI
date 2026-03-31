output "workbench_instance_names" {
  description = "作成されたWorkbenchインスタンスの名前一覧"
  value       = { for k, v in google_workbench_instance.workbench : k => v.name }
}

output "workbench_proxy_uris" {
  description = "WorkbenchインスタンスのProxy URI一覧"
  value       = { for k, v in google_workbench_instance.workbench : k => v.proxy_uri }
}
