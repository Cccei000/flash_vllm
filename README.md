# flash_vllm

- [ ] prefill.cuh里有没有accum条件 —— DTypeQ必须是half
	- [ ] 确认DTypeQKAccum的值以及和模板参数和参数列表的对应关系
- [x] 测GQA kernel的benchmark
- [ ] 看llama2-7b的latency
- [x] ToyExp没搞对