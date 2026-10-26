"""Prepared IAP route; inert until direct human authority is recorded."""
import json
from pathlib import Path
import shlex

AUTH=Path('docs/NATIVE-IAP-ACCESS-AUTHORIZATION-2026-10-26.json')
NAME='duplex-venus-us-20261026-v1'
ZONE='us-central1-a'
VM_ID='5664322998932521310'
DISK_ID='7763575121976148318'

def approved_iap_proxy():
    a=json.loads(AUTH.read_text())
    r=a['existing_resource'];b=a['unchanged_budget'];c=a['access_changes']
    if (a['status']!='APPROVED' or a['project']!='aurium-duplex-live-20261024' or
        r['vm_id']!=VM_ID or r['disk_id']!=DISK_ID or r['zone']!=ZONE or r['name']!=NAME or
        b['additional_scope_cap_usd']!=35 or b['maximum_single_restart_seconds']!=3600 or
        c['enable_only_api']!='iap.googleapis.com' or c['new_iam_principals_roles_keys_or_service_accounts'] is not False or
        c['allow_rule']['source_range']!='35.235.240.0/20' or c['allow_rule']['allow']!='tcp:22'):
        raise ValueError('Exact direct-human IAP authority required; no broader route')
    return shlex.join(['gcloud','compute','start-iap-tunnel',NAME,'22','--listen-on-stdin',
        '--zone='+ZONE,'--project=aurium-duplex-live-20261024','--verbosity=error'])
