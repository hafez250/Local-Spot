<?php
namespace Opencart\Catalog\Model\Module;

class Suggestion extends \Opencart\System\Engine\Model {
    private function getAiEndpoint(): string {
        if (defined('AI_RECOMMENDER_URL') && AI_RECOMMENDER_URL) {
            return AI_RECOMMENDER_URL;
        }

        return 'http://127.0.0.1:5000/recommend';
    }

    private function getAiTimeout(): int {
        if (defined('AI_RECOMMENDER_TIMEOUT') && (int)AI_RECOMMENDER_TIMEOUT > 0) {
            return (int)AI_RECOMMENDER_TIMEOUT;
        }

        return 2;
    }

    private function getAiCandidatePool(): int {
        if (defined('AI_RECOMMENDER_CANDIDATE_POOL') && (int)AI_RECOMMENDER_CANDIDATE_POOL >= 10) {
            return min((int)AI_RECOMMENDER_CANDIDATE_POOL, 200);
        }

        return 50;
    }

    private function getAiMinLiveEvents(): int {
        if (defined('AI_RECOMMENDER_MIN_LIVE_EVENTS') && (int)AI_RECOMMENDER_MIN_LIVE_EVENTS >= 0) {
            return min((int)AI_RECOMMENDER_MIN_LIVE_EVENTS, 200);
        }

        return 3;
    }

    private function postJson(string $url, array $payload): ?array {
        $json = json_encode($payload, JSON_UNESCAPED_SLASHES);
        if ($json === false) {
            return null;
        }

        $timeout = $this->getAiTimeout();

        if (function_exists('curl_init')) {
            $ch = curl_init($url);
            curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
            curl_setopt($ch, CURLOPT_POST, true);
            curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']);
            curl_setopt($ch, CURLOPT_POSTFIELDS, $json);
            curl_setopt($ch, CURLOPT_TIMEOUT, $timeout);

            $response = curl_exec($ch);
            $http_code = (int)curl_getinfo($ch, CURLINFO_HTTP_CODE);
            curl_close($ch);

            if ($response === false || $http_code < 200 || $http_code >= 300) {
                return null;
            }

            $data = json_decode($response, true);
            return is_array($data) ? $data : null;
        }

        $context = stream_context_create([
            'http' => [
                'method' => 'POST',
                'header' => "Content-Type: application/json\r\n",
                'content' => $json,
                'timeout' => $timeout
            ]
        ]);

        $response = @file_get_contents($url, false, $context);
        if ($response === false) {
            return null;
        }

        $data = json_decode($response, true);
        return is_array($data) ? $data : null;
    }

    private function getProductsByIds(array $product_ids): array {
        $product_ids = array_values(array_unique(array_map('intval', $product_ids)));
        if (!$product_ids) {
            return [];
        }

        $ids_sql = implode(',', $product_ids);
        $query = $this->db->query("
            SELECT p.product_id, pd.name, p.image, p.price
            FROM `" . DB_PREFIX . "product` p
            JOIN `" . DB_PREFIX . "product_description` pd ON (p.product_id = pd.product_id)
            WHERE p.product_id IN (" . $ids_sql . ")
              AND p.status = 1
              AND pd.language_id = " . (int)$this->config->get('config_language_id') . "
        ");

        $by_id = [];
        foreach ($query->rows as $row) {
            $by_id[(int)$row['product_id']] = $row;
        }

        $this->load->model('tool/image');
        $products = [];

        foreach ($product_ids as $product_id) {
            if (!isset($by_id[$product_id])) {
                continue;
            }

            $row = $by_id[$product_id];
            $products[] = [
                'product_id' => (int)$row['product_id'],
                'name'       => $row['name'],
                'thumb'      => $row['image'] ? $this->model_tool_image->resize($row['image'], 200, 200) : '',
                'price'      => $this->currency->format($row['price'], $this->session->data['currency'] ?? $this->config->get('config_currency')),
                'href'       => $this->url->link('product/product', 'product_id=' . (int)$row['product_id'])
            ];
        }

        return $products;
    }

    private function getAiSuggestedProducts(int $customer_id, int $limit = 5): array {
        $payload = [
            'user_id' => (int)$customer_id,
            'top_k' => (int)$limit,
            'candidate_k' => $this->getAiCandidatePool(),
            'min_live_events' => $this->getAiMinLiveEvents()
        ];

        $response = $this->postJson($this->getAiEndpoint(), $payload);
        if (!$response || !isset($response['recommendations']) || !is_array($response['recommendations'])) {
            return [];
        }

        $product_ids = [];
        foreach ($response['recommendations'] as $rec) {
            if (isset($rec['product_id'])) {
                $product_ids[] = (int)$rec['product_id'];
            }
        }

        return $this->getProductsByIds($product_ids);
    }

    /**
     * Get suggested products for a customer based on most purchased top-level category,
     * excluding the last product bought.
     *
     * @param int|null $customer_id If null, uses logged-in customer.
     * @param bool $debug If true, returns array with 'products' and 'debug' info (for troubleshooting).
     * @return array
     */
    public function getSuggestedProducts(int $customer_id = null, bool $debug = false, int $limit = 5): array {
        if ($customer_id === null) {
            if (!isset($this->customer) || !$this->customer->isLogged()) {
                return [];
            }
            $customer_id = (int)$this->customer->getId();
        }

        // Try AI recommender first
        if ($limit < 1) {
            $limit = 5;
        } elseif ($limit > 20) {
            $limit = 20;
        }

        $ai_products = $this->getAiSuggestedProducts($customer_id, $limit);
        if ($ai_products) {
            if ($debug) {
                return ['products' => $ai_products, 'debug' => ['source' => 'ai', 'endpoint' => $this->getAiEndpoint()]];
            }

            return $ai_products;
        }

        if ($debug) {
            return ['products' => [], 'debug' => ['reason' => 'ai_unavailable', 'endpoint' => $this->getAiEndpoint()]];
        }

        return [];
    }
}
