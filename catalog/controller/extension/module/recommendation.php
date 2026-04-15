<?php
namespace Opencart\Catalog\Controller\Extension\Module;

class Recommendation extends \Opencart\System\Engine\Controller {
    public function index(array $setting = []): string {
        if (!$this->config->get('module_recommendation_status')) {
            return '';
        }

        $this->load->language('extension/module/recommendation');
        $this->load->model('tool/image');

        if (!$this->customer->isLogged()) {
            return '';
        }

        $customer_id = $this->customer->getId();

        // Get products purchased by the customer
        $orders = $this->db->query("
            SELECT op.product_id, p2c.category_id
            FROM `" . DB_PREFIX . "order_product` op
            JOIN `" . DB_PREFIX . "order` o ON op.order_id = o.order_id
            JOIN `" . DB_PREFIX . "product_to_category` p2c ON op.product_id = p2c.product_id
            WHERE o.customer_id = '" . (int)$customer_id . "'
        ");

        if (!$orders->num_rows) {
            $data['products'] = [];
        } else {
            // Count categories
            $category_count = [];
            foreach ($orders->rows as $row) {
                $category_count[$row['category_id']] = ($category_count[$row['category_id']] ?? 0) + 1;
            }

            arsort($category_count);
            $top_category_id = key($category_count);

            // Already purchased
            $already_bought = array_column($orders->rows, 'product_id');
            $already_bought_list = implode(',', array_map('intval', $already_bought));

            $products_query = $this->db->query("
                SELECT p.product_id, pd.name, p.price, p.image, p.minimum
                FROM `" . DB_PREFIX . "product` p
                JOIN `" . DB_PREFIX . "product_description` pd ON p.product_id = pd.product_id
                JOIN `" . DB_PREFIX . "product_to_category` pc ON p.product_id = pc.product_id
                WHERE pc.category_id = '" . (int)$top_category_id . "'
                AND pd.language_id = '" . (int)$this->config->get('config_language_id') . "'
                " . ($already_bought_list ? "AND p.product_id NOT IN ($already_bought_list)" : "") . "
                LIMIT " . (int)($setting['limit'] ?? 5) . "
            ");

            $data['products'] = [];
            foreach ($products_query->rows as $product) {
                $product_data = [
                    'product_id'  => $product['product_id'],
                    'name'        => $product['name'],
                    'description' => '',
                    'thumb'       => $product['image'] ? $this->model_tool_image->resize($product['image'], 200, 200) : '',
                    'price'       => $this->currency->format($product['price'], $this->session->data['currency']),
                    'special'     => false,
                    'tax'         => false,
                    'minimum'     => (int)($product['minimum'] ?? 1),
                    'href'        => $this->url->link('product/product', 'product_id=' . $product['product_id'])
                ];

                $data['products'][] = $this->load->controller('product/thumb', $product_data);
            }
        }

        if (!$data['products']) {
            return '';
        }

        return $this->load->view('extension/module/recommendation', $data);
    }
}
