##################### Old/Unused Code (delete when final) #######################
# def warp_image(src, flow):
#   _, h, w = flow.shape
#   flow_map = np.zeros(flow.shape, dtype=np.float32)
#   for y in range(h):
#     flow_map[1,y,:] = float(y) + flow[1,y,:]
#   for x in range(w):
#     flow_map[0,:,x] = float(x) + flow[0,:,x]
#   # remap pixels to optical flow
#   dst = cv2.remap(
#     src, flow_map[0], flow_map[1], 
#     interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
#   return dst

# def get_frame(frame_num):
#     return self.frames[frame_num]

# def get_content_weights(frame, prev_frame):
#   forward_fn = args.content_weights_frmt.format(str(prev_frame), str(frame))
#   backward_fn = args.content_weights_frmt.format(str(frame), str(prev_frame))
#   forward_path = os.path.join(args.video_input_dir, forward_fn)
#   backward_path = os.path.join(args.video_input_dir, backward_fn)
#   forward_weights = read_weights_file(forward_path)
#   backward_weights = read_weights_file(backward_path)
#   return forward_weights , backward_weights

################################################################################

#TODO change all this.prev_frame to stored actual prev frame

def warp_image(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

def make_opt_flow(prev, nxt):
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    nxt_gray = cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, nxt_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def get_prev_warped_frame(frame):
    prev_frame = this.prev_frame
    # backwards flow: current frame -> previous frame
    flow = make_opt_flow(prev_frame, this_frame)
    warped_img = warp_image(prev_img, flow).astype(np.float32)
    # img = preprocess(warped_img)
    return img

#flow1 is back, flow2 is forward
def get_flow_weights(flow1, flow2): 
  xSize = flow1.shape[1]
  ySize = flow1.shape[0]
  reliable = 255 * np.ones((ySize, xSize))

  size = xSize * ySize

  x_kernel = [[-0.5, -0.5, -0.5],[0., 0., 0.],[0.5, 0.5, 0.5]]
  x_kernel = np.array(x_kernel, np.float32)
  y_kernel = [[-0.5, 0., 0.5],[-0.5, 0., 0.5],[-0.5, 0., 0.5]]
  y_kernel = np.array(y_kernel, np.float32)
  
  flow_x_dx = cv2.filter2D(flow1[:,:,0],-1,x_kernel)
  flow_x_dy = cv2.filter2D(flow1[:,:,0],-1,y_kernel)
  dx = np.stack((flow_x_dx, flow_x_dy), axis = -1)

  flow_y_dx = cv2.filter2D(flow1[:,:,0],-1,x_kernel)
  flow_y_dy = cv2.filter2D(flow1[:,:,0],-1,y_kernel)
  dy = np.stack((flow_y_dx, flow_y_dy), axis = -1)

  motionEdge = np.zeros((ySize,xSize))

  for i in range(ySize):
    for j in range(xSize): 
      motionEdge[i,j] += dy[i,j,0]*dy[i,j,0]
      motionEdge[i,j] += dy[i,j,1]*dy[i,j,1]
      motionEdge[i,j] += dx[i,j,0]*dx[i,j,0]
      motionEdge[i,j] += dx[i,j,1]*dx[i,j,1]
      

  for ax in range(xSize):
    for ay in range(ySize): 
      bx = ax + flow1[ay, ax, 0]
      by = ay + flow1[ay, ax, 1]    

      x1 = int(bx)
      y1 = int(by)
      x2 = x1 + 1
      y2 = y1 + 1
      
      if x1 < 0 or x2 >= xSize or y1 < 0 or y2 >= ySize:
        reliable[ay, ax] = 0.0
        continue 
      
      alphaX = bx - x1 
      alphaY = by - y1

      a = (1.0-alphaX) * flow2[y1, x1, 0] + alphaX * flow2[y1, x2, 0]
      b = (1.0-alphaX) * flow2[y2, x1, 0] + alphaX * flow2[y2, x2, 0]
      
      u = (1.0 - alphaY) * a + alphaY * b
      
      a = (1.0-alphaX) * flow2[y1, x1, 1] + alphaX * flow2[y1, x2, 1]
      b = (1.0-alphaX) * flow2[y2, x1, 1] + alphaX * flow2[y2, x2, 1]
      
      v = (1.0 - alphaY) * a + alphaY * b
      cx = bx + u
      cy = by + v
      u2 = flow1[ay,ax,0]
      v2 = flow1[ay,ax,1]
      
      if ((cx-ax) * (cx-ax) + (cy-ay) * (cy-ay)) >= 0.01 * (u2*u2 + v2*v2 + u*u + v*v) + 0.5: 
        # Set to a negative value so that when smoothing is applied the smoothing goes "to the outside".
        # Afterwards, we clip values below 0.
        reliable[ay, ax] = -255.0
        continue
      
      if motionEdge[ay, ax] > 0.01 * (u2*u2 + v2*v2) + 0.002: 
        reliable[ay, ax] = MOTION_BOUNDARIE_VALUE
        continue
      
  #need to apply smoothing to reliable mat
  reliable = cv2.GaussianBlur(reliable,(3,3),0)
  reliable = np.clip(reliable, 0.0, 255.0)    
  return reliable
 
def _temporal_loss(nxt, warped_prev, c):
  D = tf.size(nxt, out_type = tf.float32)
  loss = (1. / D) * tf.reduce_sum(tf.multiply(c, tf.squared_difference(nxt, warped_prev)))
  loss = tf.cast(loss, tf.float32)
  return loss

def sum_shortterm_temporal_losses(frame):
  prev_frame = this.prev_frame
  forward_flow = make_opt_flow(prev_frame, frame)
  backward_flow = make_opt_flow(frame, prev_frame)
  wp = warp_image(prev_frame)
  c = get_flow_weights(backward_flow, forward_flow)
  loss = _temporal_loss(x, wp, c)
  return loss

