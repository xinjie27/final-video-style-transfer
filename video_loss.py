def warp_image(src, flow):
  _, h, w = flow.shape
  flow_map = np.zeros(flow.shape, dtype=np.float32)
  for y in range(h):
    flow_map[1,y,:] = float(y) + flow[1,y,:]
  for x in range(w):
    flow_map[0,:,x] = float(x) + flow[0,:,x]
  # remap pixels to optical flow
  dst = cv2.remap(
    src, flow_map[0], flow_map[1], 
    interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
  return dst

def get_frame(frame_num):
    return self.frames[frame_num]

def make_opt_flow(prev, nxt):
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    nxt_gray = cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, nxt_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def get_prev_warped_frame(frame):
    this_frame = get_frame(frame)
    prev_frame = get_frame(frame-1)
    # backwards flow: current frame -> previous frame
    flow = make_opt_flow(prev_frame, this_frame)
    warped_img = warp_image(prev_img, flow).astype(np.float32)
    # img = preprocess(warped_img)
    return img

def get_content_weights(frame, prev_frame):
  forward_fn = args.content_weights_frmt.format(str(prev_frame), str(frame))
  backward_fn = args.content_weights_frmt.format(str(frame), str(prev_frame))
  forward_path = os.path.join(args.video_input_dir, forward_fn)
  backward_path = os.path.join(args.video_input_dir, backward_fn)
  forward_weights = read_weights_file(forward_path)
  backward_weights = read_weights_file(backward_path)
  return forward_weights #, backward_weights

def temporal_loss(x, w, c):
  c = c[np.newaxis,:,:,:]
  D = float(x.size)
  loss = (1. / D) * tf.reduce_sum(c * tf.nn.l2_loss(x - w))
  loss = tf.cast(loss, tf.float32)
  return loss

def sum_shortterm_temporal_losses(frame):
#   x = sess.run(net['input'].assign(input_img))
  prev_frame = frame - 1
  w = get_prev_warped_frame(frame)
  c = get_content_weights(frame, prev_frame)
  loss = temporal_loss(x, w, c)
  return loss

