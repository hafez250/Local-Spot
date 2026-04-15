<?php
namespace Opencart\Catalog\Model\Tool;

class AiEvent extends \Opencart\System\Engine\Model {
    private function getAiInteractEndpoint(): string {
        if (defined('AI_INTERACT_URL') && AI_INTERACT_URL) {
            return AI_INTERACT_URL;
        }

        if (defined('AI_RECOMMENDER_URL') && AI_RECOMMENDER_URL) {
            return str_replace('/recommend', '/interact', AI_RECOMMENDER_URL);
        }

        return 'http://127.0.0.1:5000/interact';
    }

    private function getAiTimeout(): int {
        if (defined('AI_RECOMMENDER_TIMEOUT') && (int)AI_RECOMMENDER_TIMEOUT > 0) {
            return (int)AI_RECOMMENDER_TIMEOUT;
        }

        return 2;
    }

    private function postJson(string $url, array $payload): bool {
        $json = json_encode($payload, JSON_UNESCAPED_SLASHES);
        if ($json === false) {
            return false;
        }

        if (function_exists('curl_init')) {
            $ch = curl_init($url);
            curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
            curl_setopt($ch, CURLOPT_POST, true);
            curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']);
            curl_setopt($ch, CURLOPT_POSTFIELDS, $json);
            curl_setopt($ch, CURLOPT_TIMEOUT, $this->getAiTimeout());
            curl_exec($ch);
            $http_code = (int)curl_getinfo($ch, CURLINFO_HTTP_CODE);
            curl_close($ch);

            return $http_code >= 200 && $http_code < 300;
        }

        $context = stream_context_create([
            'http' => [
                'method' => 'POST',
                'header' => "Content-Type: application/json\r\n",
                'content' => $json,
                'timeout' => $this->getAiTimeout()
            ]
        ]);

        $result = @file_get_contents($url, false, $context);
        return $result !== false;
    }

    private function pushToAi(int $customer_id, int $product_id, string $event_type): void {
        if ($customer_id <= 0) {
            return;
        }

        $api_event_type = $event_type === 'add_to_cart' ? 'addtocart' : $event_type;
        $payload = [
            'user_id' => $customer_id,
            'product_id' => $product_id,
            'event_type' => $api_event_type
        ];

        $this->postJson($this->getAiInteractEndpoint(), $payload);
    }

    public function logEvent(int $product_id, string $event_type): void {
        if ($product_id <= 0) {
            return;
        }

        $allowed = ['view', 'add_to_cart', 'purchase'];
        if (!in_array($event_type, $allowed, true)) {
            return;
        }

        $customer_id = 0;
        if (isset($this->customer) && $this->customer->isLogged()) {
            $customer_id = (int)$this->customer->getId();
        }

        $session_id = '';
        if (isset($this->session) && method_exists($this->session, 'getId')) {
            $session_id = (string)$this->session->getId();
        }

        $this->db->query("
            INSERT INTO `" . DB_PREFIX . "ai_event`
            SET customer_id = '" . (int)$customer_id . "',
                session_id  = '" . $this->db->escape($session_id) . "',
                product_id  = '" . (int)$product_id . "',
                event_type  = '" . $this->db->escape($event_type) . "',
                date_added  = NOW()
        ");

        $this->pushToAi($customer_id, $product_id, $event_type);
    }
}
