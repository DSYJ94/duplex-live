"""The revised recorder must preserve raw samples and reject fixture approval."""
from evals.harness import test_native_reasoning_teacher as original
from evals import native_reasoning_teacher_v2 as revised
from evals.native_reasoning_stream_v2 import capture


class RevisedArtifacts(original.TeacherArtifacts):
    implementation=revised
    capture_stream=staticmethod(capture)


if __name__=='__main__':__import__('unittest').main()
