"""Scheduling correctness only: no provider, benchmark data or efficacy run."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('parallel_regional',scripts/'parallel_regional_structural_matrix.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_worker_control_is_bounded_and_live(tmp_path):
    control=tmp_path/'WORKERS.txt'
    assert m.worker_limit(control,2)==2
    control.write_text('4')
    assert m.worker_limit(control,2)==4
    control.write_text('5')
    with pytest.raises(ValueError): m.worker_limit(control,2)


def test_child_is_isolated_and_receives_thread_limit(tmp_path,monkeypatch):
    row={'task':'synthetic','seed':7}
    env={key:'1' for key in m.THREAD_ENV}
    def run(command,**kwargs):
        assert kwargs['env']==env and kwargs['timeout']==42
        assert command[-2:]==['--coordinate-index','3']
        child=tmp_path/'worker_003'
        child.mkdir()
        m.write(child/'PROGRESS_003.json',{'coordinate':row,'failure':'FixtureFailure','message':'synthetic'})
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(m.subprocess,'run',run)
    result=m.execute(3,row,tmp_path,tmp_path/'FREEZE.json',42,env)
    assert result['failure']=='FixtureFailure'
    assert result['artifact_root']==str(tmp_path/'worker_003'/'coordinate_003')
    assert (tmp_path/'worker_003.log').exists()


def test_timeout_is_terminal_and_success_cannot_hide_invalid_artifacts(tmp_path,monkeypatch):
    def timeout(*a,**kw): raise m.subprocess.TimeoutExpired(a[0],5)
    monkeypatch.setattr(m.subprocess,'run',timeout)
    row={'task':'synthetic','seed':7}
    assert m.execute(0,row,tmp_path,tmp_path/'FREEZE.json',5,{})['failure']=='CoordinateTimeout'
    with pytest.raises(ValueError,match='coordinate identity'):
        m.verify_record({'coordinate':{'task':'other'},'failure':'x'},row,tmp_path)
    with pytest.raises(FileNotFoundError):
        m.verify_record({'coordinate':row,'result':{'protocol_complete':True}},row,tmp_path)
