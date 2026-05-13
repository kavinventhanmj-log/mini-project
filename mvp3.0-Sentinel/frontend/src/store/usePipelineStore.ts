import { create } from 'zustand';

interface PipelineState {
  isRunning: boolean;
  isDone: boolean;
  totalProcessed: number;
  reviewCount: number;
  setStatus: (status: { pipeline_running: boolean; pipeline_done: boolean; total_processed: number; review_queue: number }) => void;
}

export const usePipelineStore = create<PipelineState>((set) => ({
  isRunning: false,
  isDone: false,
  totalProcessed: 0,
  reviewCount: 0,
  setStatus: (status) => set({
    isRunning: status.pipeline_running,
    isDone: status.pipeline_done,
    totalProcessed: status.total_processed,
    reviewCount: status.review_queue,
  }),
}));
