def custom_loss_wrapper(normFac=1):
    '''
    customized loss function to improve the recoil response,
    by balancing the response above one and below one
    '''


    def custom_loss(y_true, y_pred):
        import tensorflow.keras.backend as K
        import tensorflow as tf

        px_truth = K.flatten(y_true[:, 0])
        py_truth = K.flatten(y_true[:, 1])
        px_pred = K.flatten(y_pred[:, 0])
        py_pred = K.flatten(y_pred[:, 1])
        eps = K.epsilon()
        pt_truth = K.sqrt(px_truth*px_truth + py_truth*py_truth)

        #px_truth1 = px_truth / pt_truth
        #py_truth1 = py_truth / pt_truth

        # using absolute response
        # upar_pred = (px_truth1 * px_pred + py_truth1 * py_pred)/pt_truth

        arg = px_pred * px_pred + py_pred * py_pred
        eps = tf.cast(K.epsilon(), arg.dtype)
        arg = tf.maximum(arg, eps)
        upar_pred = K.sqrt(arg) - pt_truth


        pt_cut = pt_truth > 0.
        upar_pred = tf.boolean_mask(upar_pred, pt_cut)
        pt_truth_filtered = tf.boolean_mask(pt_truth, pt_cut)

        # upar_pred = K.sqrt(px_pred * px_pred + py_pred * py_pred) - pt_truth
        # pt_cut = pt_truth > 0.
        # upar_pred = tf.boolean_mask(upar_pred, pt_cut)
        # pt_truth_filtered = tf.boolean_mask(pt_truth, pt_cut)
        #filter_bin0 = pt_truth_filtered < 50./normFac
        filter_bin0 = tf.logical_and(pt_truth_filtered > 50./normFac,  pt_truth_filtered < 100./normFac)
        filter_bin1 = tf.logical_and(pt_truth_filtered > 100./normFac, pt_truth_filtered < 200./normFac)
        filter_bin2 = tf.logical_and(pt_truth_filtered > 200./normFac, pt_truth_filtered < 300./normFac)
        filter_bin3 = tf.logical_and(pt_truth_filtered > 300./normFac, pt_truth_filtered < 400./normFac)
        filter_bin4 = pt_truth_filtered > 400./normFac

        upar_pred_pos_bin0 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin0, upar_pred > 0.))
        upar_pred_neg_bin0 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin0, upar_pred < 0.))
        upar_pred_pos_bin1 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin1, upar_pred > 0.))
        upar_pred_neg_bin1 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin1, upar_pred < 0.))
        upar_pred_pos_bin2 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin2, upar_pred > 0.))
        upar_pred_neg_bin2 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin2, upar_pred < 0.))
        upar_pred_pos_bin3 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin3, upar_pred > 0.))
        upar_pred_neg_bin3 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin3, upar_pred < 0.))
        upar_pred_pos_bin4 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin4, upar_pred > 0.))
        upar_pred_neg_bin4 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin4, upar_pred < 0.))
        #upar_pred_pos_bin5 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin5, upar_pred > 0.))
        #upar_pred_neg_bin5 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin5, upar_pred < 0.))
        norm = tf.reduce_sum(pt_truth_filtered)
        dev = tf.abs(tf.reduce_sum(upar_pred_pos_bin0) + tf.reduce_sum(upar_pred_neg_bin0))
        dev += tf.abs(tf.reduce_sum(upar_pred_pos_bin1) + tf.reduce_sum(upar_pred_neg_bin1))
        dev += tf.abs(tf.reduce_sum(upar_pred_pos_bin2) + tf.reduce_sum(upar_pred_neg_bin2))
        dev += tf.abs(tf.reduce_sum(upar_pred_pos_bin3) + tf.reduce_sum(upar_pred_neg_bin3))
        dev += tf.abs(tf.reduce_sum(upar_pred_pos_bin4) + tf.reduce_sum(upar_pred_neg_bin4))
        #dev += tf.abs(tf.reduce_sum(upar_pred_pos_bin5) + tf.reduce_sum(upar_pred_neg_bin5))
        dev /= norm + eps

        loss = 0.5*normFac**2*K.mean((px_pred - px_truth)**2 + (py_pred - py_truth)**2)

        #loss += 200.*dev
        loss += 7000.*dev
        return loss
        
    return custom_loss


def hybrid_vector_recoil_loss(normFac=1.0, w_vec=1.0, w_recoil=5000.0, eps=1e-6, dbg=False):
    import tensorflow as tf

    huber = tf.keras.losses.Huber(delta=1.0, reduction="none")

    def _finite(x, fill=0.0):
        return tf.where(tf.math.is_finite(x), x, tf.fill(tf.shape(x), tf.cast(fill, x.dtype)))

    def loss_fn(y_true, y_pred):
        # 1) components
        px_t, py_t = y_true[:, 0], y_true[:, 1]
        px_p, py_p = y_pred[:, 0], y_pred[:, 1]

        # 2) magnitudes (safe)
        pt_t = tf.sqrt(tf.maximum(px_t*px_t + py_t*py_t, eps))
        pt_p = tf.sqrt(tf.maximum(px_p*px_p + py_p*py_p, eps))
        pt_t = _finite(pt_t); pt_p = _finite(pt_p)

        # 3) vector-based huber
        # (a) magnitude
        loss_mag = huber(pt_t, pt_p)
        loss_mag = _finite(loss_mag)

        # (b) angle
        denom = tf.maximum(pt_t * pt_p, eps)
        cos_theta = (px_t*px_p + py_t*py_p) / denom
        cos_theta = tf.clip_by_value(cos_theta, -1.0, 1.0)
        cos_theta = _finite(cos_theta, fill=0.0)
        loss_ang = huber(tf.ones_like(cos_theta), cos_theta)
        loss_ang = _finite(loss_ang)

        # (c) vector difference
        vec_diff = tf.sqrt(tf.maximum((px_p - px_t)**2 + (py_p - py_t)**2, eps))
        vec_diff = _finite(vec_diff)
        loss_vec = huber(tf.zeros_like(vec_diff), vec_diff)
        loss_vec = _finite(loss_vec)

        # (d) dot difference
        dot_diff = (px_p * px_t + py_p * py_t) - pt_t * pt_t
        dot_diff = _finite(dot_diff)
        loss_dot = huber(tf.zeros_like(dot_diff), abs(dot_diff))
        loss_dot = _finite(loss_dot)

        w_loss_vec, w_loss_mag, w_loss_ang = 1, 0.5, 0.2

        vector_loss = tf.reduce_mean(w_loss_vec * loss_vec + w_loss_mag * loss_mag + w_loss_ang * loss_ang)
        # vector_loss = tf.reduce_mean(loss_vec + loss_mag + loss_ang + loss_dot)
        vector_loss = _finite(vector_loss)
        
        # vector_loss = tf.reduce_mean(loss_dot)
        # vector_loss = _finite(vector_loss)

        # vector_loss = huber(pt_t**2 + pt_t, cos_theta * pt_t * pt_p + pt_p)
        # vector_loss = _finite(vector_loss)

        # 4) recoil-balance (bin-wise)
        # bins = [
        #     (50./normFac, 100./normFac),
        #     (100./normFac, 150./normFac),
        #     (150./normFac, 200./normFac),
        #     (200./normFac, 250./normFac),
        #     (250./normFac, 300./normFac),
        #     (300./normFac, 400./normFac),
        #     (400./normFac, 9999.)
        # ]

        bins = [
            (50./normFac, 70./normFac, 1.0),
            (70./normFac, 90./normFac, 1.0),
            (90./normFac, 110./normFac, 1.0),
            (110./normFac, 130./normFac, 1.0),
            (130./normFac, 150./normFac, 1.0),
            (150./normFac, 170./normFac, 1.0),
            (170./normFac, 250./normFac, 1.0),
            (250./normFac, 300./normFac, 1.0),
            (300./normFac, 400./normFac, 1.0),
            (400./normFac, 9999., 1.0)
        ]

        upar_pred = pt_p - pt_t
        mask = pt_t > 0.0
        upar_pred = tf.boolean_mask(upar_pred, mask)
        pt_t_f = tf.boolean_mask(pt_t, mask)

        dev_sum = tf.constant(0.0, dtype=tf.float32)
        bias_sum = tf.constant(0.0, dtype=tf.float32)
        width_sum = tf.constant(0.0, dtype=tf.float32)

        c = tf.cast(20.0/normFac, tf.float32)

        for bmin, bmax, w in bins:
            in_bin = tf.logical_and(pt_t_f > bmin, pt_t_f < bmax)
            u = tf.boolean_mask(upar_pred, in_bin)
            pt_bin = tf.boolean_mask(pt_t_f, in_bin)

            dev_sum += w * tf.cond(
                tf.size(u) > 0,
                lambda: tf.abs(tf.reduce_sum(tf.boolean_mask(u, u > 0.0)) +
                               tf.reduce_sum(tf.boolean_mask(u, u < 0.0))),
                lambda: tf.constant(0.0, dtype=tf.float32)
            )

            bias_sum += w * tf.cond(
                tf.size(u) > 0,
                lambda: tf.abs(tf.reduce_mean(u)),
                lambda: tf.constant(0.0, dtype=tf.float32)
            )

            u_rel = u / (pt_bin + c)
            width_sum += w * tf.cond(
                tf.size(u_rel) > 0,
                lambda: tf.reduce_mean(huber(tf.zeros_like(u_rel), u_rel)),
                lambda: tf.constant(0.0, dtype=tf.float32)
            )

        norm = tf.reduce_sum(pt_t_f) + eps
        # norm = 1
        recoil_dev = _finite(dev_sum / norm)
        recoil_bias = _finite(bias_sum / norm)
        recoil_width = _finite(width_sum / tf.cast(len(bins), tf.float32))  # average over bins


        # u = pt_p - pt_t
        # mask_hi = pt_t > 200.0
        # under_rel = tf.boolean_mask(tf.nn.relu(-u) / (pt_t + 20.0), mask_hi)

        # k = tf.maximum(1, tf.cast(0.05 * tf.cast(tf.size(under_rel), tf.float32), tf.int32))
        # loss_tail = tf.reduce_mean(tf.nn.top_k(under_rel, k=k, sorted=False).values)

        # under_rel = tf.boolean_mask(...)

        # n = tf.size(under_rel)

        # def topk_loss():
        #     k = tf.maximum(1, tf.cast(0.05 * tf.cast(n, tf.float32), tf.int32))
        #     vals = tf.nn.top_k(under_rel, k=k, sorted=False).values
        #     return tf.reduce_mean(vals)

        # loss_tail = tf.cond(n > 0, topk_loss, lambda: tf.constant(0.0, tf.float32))
        # loss += w_tail * loss_tail
        # if dbg:
        #     tf.print("VL=", vector_loss, "RD=", recoil_dev,
        #              "pt_t[min,max,mean]=", tf.reduce_min(pt_t), tf.reduce_max(pt_t), tf.reduce_mean(pt_t),
        #              summarize=6)

        # loss = w_vec * vector_loss + w_recoil * recoil_dev
        loss = w_vec * vector_loss + 5000 * (recoil_dev + recoil_bias + recoil_width)
        # loss = w_vec * vector_loss + w_recoil * recoil_dev + 5000 * (recoil_bias + recoil_width) + 10000 * loss_tail
        return _finite(loss)

    return loss_fn

def deepmet_like_loss(normFac=1.0, loss_type="huber"):
    import tensorflow as tf

    if loss_type == "mse":
        def loss_fn(y_true, y_pred):
            dx = y_pred[:, 0] - y_true[:, 0]
            dy = y_pred[:, 1] - y_true[:, 1]
            return 0.5 * tf.reduce_mean(dx * dx + dy * dy)
        return loss_fn

    huber = tf.keras.losses.Huber(delta=1.0, reduction="none")

    def loss_fn(y_true, y_pred):
        loss_x = huber(y_true[:, 0], y_pred[:, 0])
        loss_y = huber(y_true[:, 1], y_pred[:, 1])
        return tf.reduce_mean(loss_x + loss_y)

    return loss_fn