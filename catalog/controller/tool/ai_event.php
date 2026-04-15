<?php
namespace Opencart\Catalog\Controller\Tool;

class AiEvent extends \Opencart\System\Engine\Controller {
    public function log(): void {
        $this->response->addHeader('Content-Type: application/json');

        if ($this->request->server['REQUEST_METHOD'] !== 'POST') {
            $this->response->setOutput(json_encode(['ok' => false, 'error' => 'method_not_allowed']));
            return;
        }

        $product_id = (int)($this->request->post['product_id'] ?? 0);
        $event_type = (string)($this->request->post['event_type'] ?? '');

        if ($product_id <= 0 || $event_type !== 'view') {
            $this->response->setOutput(json_encode(['ok' => false, 'error' => 'invalid']));
            return;
        }

        $this->load->model('tool/ai_event');
        $this->model_tool_ai_event->logEvent($product_id, 'view');

        $this->response->setOutput(json_encode(['ok' => true]));
    }
}
